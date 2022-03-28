import os
import sys
import copy
import configparser
import numpy as np
import pandas as pd
from scipy import interpolate
from tools.hyades_runner import batch_run_hyades
from tools.hyades_reader import HyadesOutput, ShockVelocity


def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    total = len(iterable)

    def printProgressBar(iteration):
        """Progress Bar Printing Function"""
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        sys.stdout.write(f'\r\t{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


class Jacobian:

    def __init__(self, run_name, n=0):
        """Set up run_name, experimental data, optimization_json, and config file"""
        self.run_name = run_name
        self.inf_directory = os.path.join('.', 'data', 'inf')
        self.path = os.path.join('.', 'data', self.run_name)

        config_filename = os.path.join(self.path, f'{run_name}.cfg')
        config = configparser.ConfigParser()
        config.read(config_filename)
        self.time = np.array([float(i) for i in config.get('Setup', 'time').split(', ')])
        if n > 0:
            self.time = np.linspace(self.time.min(), self.time.max(), num=n)

        self.initial_pressure = np.array([float(i) for i in config.get('Setup', 'pressure').split(',')])
        # Load experimental data
        experimental_filename = config.get('Experimental', 'filename',
                                           fallback=self.run_name)
        if experimental_filename == 'None':
            experimental_filename = self.run_name
        if not experimental_filename.endswith('.xlsx'):
            experimental_filename += '.xlsx'
        df = pd.read_excel(os.path.join('.', 'data', 'experimental', experimental_filename), sheet_name=0)
        df.loc[df[df.columns[1]] < 0.1, df.columns[1]] = 0  # set velocities below 0.1 to 0
        experimental_time = df[df.columns[0]]
        experimental_velocity = df[df.columns[1]]
        '''
        Excel adds NaN rows to Velocity Time and Velocity if the other columns in the sheet are longer
        This drops all rows from velocity and time if the first instance of NaN till the end of the file is NaN
        if it can't easily remove the NaNs it raises an error
        '''
        if any(experimental_velocity.isna()):
            index_of_first_nan = min(np.where(experimental_velocity.isna())[0])
            if all(experimental_time[index_of_first_nan:].isna()):
                experimental_time = experimental_time[:index_of_first_nan]
                experimental_velocity = experimental_velocity[:index_of_first_nan]
            else:
                raise ValueError(f'Found NaN (Not-a-Number) in {self.exp_file} and could not easily remove them.\n'
                                 f'This might be caused by blank rows or invalid numbers in {self.exp_file}')
        '''
        PyHy Optimizer interpolates experimental time onto 50 evenly space points.
        This timing is also used for the residual calculation.
        '''
        f = interpolate.interp1d(experimental_time, experimental_velocity)
        self.experimental_time = np.linspace(min(experimental_time), max(experimental_time), num=100)
        self.experimental_velocity = f(self.experimental_time)

    def calculate_jacobian(self, pressure, delta=10):
        """Returns the partial derivative in each dimension using Hyades simulations"""
        partial_derivatives = []
        for i, p in enumerate(progressBar(pressure, prefix='Jacobian:', suffix='Calculated', length=40)):
            new_pressure = copy.deepcopy(pressure)
            new_pressure[i] += delta
            inf_name = self.write_inf(i, new_pressure)
            # print(inf_name)
            batch_run_hyades(self.inf_directory, self.path, quiet=True)
            partial_derivatives.append(self.calculate_residual(inf_name))

            # Remove Hyades run to save space
            ith_jacobian= f'{self.run_name}_jacobian{i}'
            for file_extension in ('.inf', '.otf', '.ppf', '.tmf', '.cdf'):
                filename = os.path.join(self.path, ith_jacobian, ith_jacobian + file_extension)
                if os.path.exists(filename):
                    os.remove(filename)

            try:
                os.rmdir(os.path.join(self.path, ith_jacobian))
            except OSError as e:
                print(f'Failed to delete the directory {ith_jacobian} - Check remaining contents:')
                print(os.listdir(os.path.join(self.path, ith_jacobian)))
                raise e
        print('\tJacobian:', ', '.join([f'{i:.2f}' for i in partial_derivatives]))
        return partial_derivatives

    def write_inf(self, i, pressure):
        """Write an inf for each point in pressure

        Args:
            i:
            pressure:

        Returns:
            inf_names (list): the names of each inf written
        """
        # Interpolate time and pressure to smooth out linear segments
        f = interpolate.interp1d(self.time, pressure, kind='cubic')
        interpolated_time = np.linspace(np.ceil(self.time.min()), np.floor(self.time.max()), num=100)
        interpolated_pressure = f(interpolated_time)
        # Format pressure for Hyades inf
        pressure_lines = [f'tv {t * 1e-9:.6e} {v * 1e10:.6e}'
                          for t, v in zip(interpolated_time, interpolated_pressure)]
        # Read the setup.inf for formatting
        setup_inf_filename = f'{self.run_name}_setup.inf'
        with open(os.path.join(self.path, setup_inf_filename)) as f:
            lines = f.read()
        assert 'TV_PRES' in lines, f'Could not find TV_PRES in {setup_inf_filename}'
        # Replace the TV_PRES keyword in the setup.inf with the new pressure and write the new inf
        new_lines = lines.replace('TV_PRES', '\n'.join(pressure_lines))
        inf_name = setup_inf_filename.replace('setup', f'jacobian{i}')
        with open(os.path.join(self.inf_directory, inf_name), 'w') as f:
            f.write(new_lines)

        return inf_name

    def calculate_residual(self, inf_name):
        """Computes the sum of squares difference between a completed Hyades run and experimental data

        Note:
            Currently this is not written for shock velocities

        Args:
            inf_name:

        Returns:
            residual (numpy array): An array of the partial derivative in each dimension
        """
        if inf_name.endswith('.inf'):
            inf_name = inf_name[:-4]
        f = os.path.join(self.path, inf_name, inf_name)
        hyades = HyadesOutput(f, 'U')
        last_zone_index = hyades.layers[hyades.moi]['Mesh Stop'] - 1

        f = interpolate.interp1d(hyades.time, hyades.output[:, last_zone_index])
        interpolated_hyades_u = f(self.experimental_time)

        return sum(np.square(self.experimental_velocity - interpolated_hyades_u))

    def run_in_parallel(self, inf_names):
        """Runs all inf_names in parallel

        Args:
            inf_names:
        """
        pass


if __name__ == '__main__':
    jac = Jacobian('220303_s77731')
    print(jac.run_name, jac.path)
    print(jac.inf_directory)
    print(jac.time, jac.initial_pressure)
    # plt.plot(jac.experimental_time, jac.experimental_velocity)
    # plt.show()
    jacobian = jac.calculate_jacobian(jac.initial_pressure)
    print(jacobian)