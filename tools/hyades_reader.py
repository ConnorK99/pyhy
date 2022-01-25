import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
from scipy.interpolate import CubicSpline


class HyadesOutput:
    """Gets and stores Hyades simulation info from the .inf and .cdf

    This class is for use with Hyades .cdf files generated by the function PPF2NCDF. It stores many of the inputs
    specified in the .inf file along with many of the computed results from a .cdf file for a given variable of interest

    Note:
        Hyades uses two computation grids, the Zones and the Mesh.
        The Mesh points are the boundaries between Zones, drawn below, where each | is a Mesh point.
                                        | Zone_1 | Zone_2 | Zone 3 |
                                      mesh_1   mesh_2   mesh_3   mesh_4
        Which means there is always one more mesh point than there are zones.
        Some variables, such as Particle Velocity, place Lagrangian coordinates at Mesh points.
        Other variables, such as Pressure, Density, and Temperature, place Lagrangian coordinates in Zones.
        Zone coordinates are computed as the average of their left and right Mesh points.

    Attributes:
        filename (string): Name used to initialize
        dir_name (string): Name of containing directory
        run_name (string): Name of the Hyades run with no file extension or directories
        var (string): Abbreviated name of the variable of interest used to init
        x (numpy array): Lagrangian coordinates of the simulation in microns
        time (numpy array): Times of the simulation in nanoseconds
        output (numpy array): Variable specified by var. For most variables is a 2D array with len(time) rows and
                              len(x) columns.
        long_name (string): Full name of var according to Hyades
        units (string): SI units for the variable of interest
        layers (dict): Dictionary of the layers and their properties specified by the mesh line in the .inf
        moi (string): Material of interest if one is selected, otherwise None
        shock_moi (string): Shock material of interest if one is selected, otherwise None
        tv (dict): Dictionary of all drives in the inf. May include each of Pressure, Temperature, Laser drives.
        xray_probe (tuple): Tuple of (xray_start_time, xray_stop_time) if specified, otherwise None

    """
    def __init__(self, filename, var):
        """Gets and stores Hyades simulation info from the .inf and .cdf

        Args:
            filename (string): Name of the .inf (does not require file extension)
            var (string): Abbreviated name of variable of interest - one of Pres, Rho, U, Te, Ti, Tr, R

        """
        self.filename = filename
        if os.path.isdir(filename):
            self.dir_name = filename
        elif os.path.isdir(os.path.join('.', 'data', filename)):
            self.dir_name = os.path.join('.', 'data', filename)
        else:
            self.dir_name = os.path.dirname(filename)

        self.run_name = os.path.splitext(os.path.basename(filename))[0]
        self.var = var.capitalize()

        # Get variable information from cdf
        if self.run_name + '.cdf' in os.listdir(self.dir_name):
            cdf_name = os.path.join(self.dir_name, self.run_name+'.cdf')
            x, time, output, long_name, units, data_dimensions = self.get_var_from_cdf(cdf_name, self.var)
        else:
            raise Exception(f"Could not find {self.run_name+'.cdf'} in {self.dir_name}")
        self.x = x
        self.time = time
        self.output = output
        self.long_name = long_name
        self.units = units
        self.data_dimensions = data_dimensions

        # Get layer information from .inf
        if self.run_name + '.inf' in os.listdir(self.dir_name):
            inf_name = os.path.join(self.dir_name, self.run_name + '.inf')
            layers, moi, shock_moi = self.get_layers(inf_name)
            tv = self.get_tv(inf_name)
            xray_probe = self.get_xray_time(inf_name)
        else:
            raise Exception(f"Could not find {self.run_name + '.inf.'} in {self.dir_name}")
        self.layers = layers
        self.moi = moi
        self.shock_moi = shock_moi
        self.tv = tv
        self.xray_probe = xray_probe

    @staticmethod
    def get_var_from_cdf(filename, var):
        """Reads the time, Lagrangian position, and a single variable from a .cdf

        Args:
            filename (string): Name of the .inf
            var (string): Abbreviated name of variable of interest - one of Pres, Rho, U, Te, Ti, Tr, R

        Returns:
            x (numpy array): Lagrangian coordinates of the simulation in microns
            time (numpy array): Times of the simulation in nanoseconds
            output (numpy array): Variable specified by var. For most variables is a 2D array with len(time) rows and
                                  len(x) columns.
            long_name (string): Full name of var according to Hyades
            units (string): SI units for the variable of interest

        """
        cdf = netcdf.netcdf_file(filename, 'r')
        time = cdf.variables['DumpTimes'].data.copy() * 1e9  # convert seconds to nanoseconds
        x = cdf.variables['R'].data.copy() * 1e4  # x is the mesh coordinates, convert cm to um
        output = cdf.variables[var].data.copy()  # output may be a 1D, 2D, or 3D array depending on the variable
        data_dimensions = cdf.variables[var].dimensions
        # FIXME: what does sd1 do with the pressure calculations. Ray thinks it needs to be subtracted
        # if var == 'Pres':
        #     sd1 = cdf.variables['Sd1'].data.copy() * 1e-10
        #     output = output - sd1

        if cdf.variables[var].dimensions[1] == 'NumMeshs':
            pass
        elif cdf.variables[var].dimensions[1] == 'NumZones':
            x = (x[:, 1:] + x[:, :-1]) / 2
        else:
            raise Exception(f'Unexpected size of {var!r} data array: {cdf.variables[var].dimensions}')

        long_name = cdf.variables[var].long_name.decode('utf-8')
        units = cdf.variables[var].units.decode('utf-8')

        cdf.close()
        '''
        All conversions below change the Hyades default cgs units to SI units
        Most conversions taken from https://en.wikipedia.org/wiki/Centimetre%E2%80%93gram%E2%80%93second_system_of_units
        unit_conversions that are commented out have not been confirmed
        '''
        if 'Acc' == var:
            long_name = 'Mesh Acceleration'
            units = 'km/s^2'
            unit_conversion = 1e-5
        elif 'Akappa' == var:
            unit_conversion = 1  # 0.1
        elif 'Conde' == var:
            unit_conversion = 1  # 8.62e-13
        elif 'Condi' == var:
            unit_conversion = 1  # 8.62e-13
        elif 'Eelc' == var:
            units = 'Joules'
            unit_conversion = 1e-7
        elif 'Eion' == var:
            units = 'Joules'
            unit_conversion = 1e-7
        elif 'Ekappa' == var:
            unit_conversion = 1  # 0.1
        elif 'Pres' == var:
            long_name = 'Pressure'
            units = 'GPa'
            unit_conversion = 1e-10
        elif 'Qrad' == var:
            units = 'Watts/m^2'
            unit_conversion = 1e-3
        elif 'Qradgl' == var:
            unit_conversion = 1  # 8.62e-11
        elif 'Qradgr' == var:
            unit_conversion = 1  # 8.62e-11
        elif 'R' == var:
            long_name = 'Eulerian Position'
            units = 'µm'
            unit_conversion = 1e4
        elif 'RCM' == var:
            long_name = 'Eulerian Zone Position'
            units = 'µm'
            unit_conversion = 1e4
        elif 'Rho' == var:
            long_name = 'Density'
            units = 'g/cc'
            unit_conversion = 1
        elif 'Sd1' == var:
            units = 'GPa'
            unit_conversion = 1e-10
        elif var in ('Seelc', 'Seion', 'Serad'):
            units = 'Joules'
            unit_conversion = 1e-7
        elif var in ('Te', 'Ti', 'Tr'):
            units = '° K'
            unit_conversion = 11604 * 1000
        elif 'U' == var:
            long_name = 'Particle Velocity'
            units = 'km/s'
            unit_conversion = 1e-5
        elif 'Ucm' == var:
            long_name = 'Zone Particle Velocity'
            units = 'km/s'
            unit_conversion = 1e-5
        elif 'Ubin' == var:
            units = 'Joules/(K * m^3)'
            unit_conversion = 8.62e-9
        else:
            raise InvalidVariable(f'HyadesOutput does not recognize variable: {var}')

        output *= unit_conversion

        return x, time, output, long_name, units, data_dimensions

    def get_closest_time(self, requested_time):
        """Get the closest time and its index output by Hyades

        Args:
            requested_time (float):

        Returns:
            closest_time (float), index_of_closest_time (int)

        """
        index = np.argmin(abs(self.time - requested_time))
        closest_time = self.time[index]
        return closest_time, index

    @staticmethod
    def get_tv(filename):
        """Gets all tv inputs specified in the .inf

        Args:
             filename (string): Name of the .inf

        Returns:
            tv (dict): All times, in nanoseconds, and values, SI units, for each source input

        """
        if not filename.endswith('.inf'):
            filename += '.inf'
        with open(filename) as f:
            lines = f.readlines()

        # The format of the .inf files dictates a source line, then an optional sourcem line, then multiple tv lines
        tv = {}
        mode = None
        sourcem = 1
        unit_conversion = 1
        for line in lines:
            if line.startswith('source '):
                mode = line.split()[1]
                tv[f'{mode}-t'] = []
                tv[f'{mode}-v'] = []
                if mode.lower() == 'pres':
                    unit_conversion = 1e-10
                elif mode.lower() == 'te':
                    unit_conversion = 11605 * 1000
                elif mode.lower() == 'laser':  # unsure of laser units, they might depend on Hyades geometry
                    unit_conversion = 1

            if line.startswith('sourcem '):
                sourcem = float(line.split()[1])

            if line.startswith('tv '):
                t = float(line.split()[1]) * 1e9  # convert seconds to nanoseconds
                v = float(line.split()[2]) * sourcem * unit_conversion  # apply source mulitplier and convert to SI units
                tv[f'{mode}-t'].append(t)
                tv[f'{mode}-v'].append(v)

        return tv

    @staticmethod
    def get_xray_time(filename):
        """Pulls the X-Ray Probe start and stop time from the .inf, if it's in the comments

        Args:
            filename (string): Name of the .inf

        Returns:
            xray_probe (tuple): (xray_start, xray_stop) times in nanoseconds if found, otherwise None

        """
        with open(filename) as f:
            lines = f.readlines()
        xray_line = [line for line in lines if line.startswith('c xray_probe ')]
        if xray_line:
            xray_line = xray_line[0]
            xray_start = float(xray_line.split()[2])
            xray_stop = float(xray_line.split()[3])
            xray_probe = tuple((xray_start, xray_stop))
            return xray_probe
        else:
            return None

    @staticmethod
    def get_layers(filename):
        """Gets target information and material of interest from a .inf

        Args:
            filename (string): Name of the .inf

        Returns:
            layers (dict): Name, EOS number, mesh properties, and initial positions of each layer

        """
        inf_name = filename
        if inf_name.endswith('.cdf'):
            inf_name = inf_name[:-4]
        if not inf_name.endswith('.inf'):
            inf_name += '.inf'

        with open(filename) as f:
            lines = f.readlines()

        eos_lines = [line for line in lines if line.startswith('EOS ')]
        mesh_lines = [line for line in lines if line.startswith('mesh ')]

        pattern = '\[\w+!?\$?\]'
        result = re.findall(pattern, ''.join(lines))

        assert(len(result) == len(eos_lines)), f'Unequal number of material and EOS lines.' \
                                               f'\nMaterials: {result}\nEOS: {eos_lines}'
        assert(len(result) == len(mesh_lines)), f'Unequal number of material and mesh lines' \
                                                f'\nMaterials: {result}\nMesh Lines: {mesh_lines}'

        layers = {}
        material_of_interest = None
        shock_material_of_interest = None
        for i in range(len(result)):
            k = f'layer{i+1}'
            layers[k] = {}
            bare_name = result[i][1:-1].replace('!', '').replace('$', '')
            if '!' in result[i]:
                material_of_interest = k
            if '$' in result[i]:
                shock_material_of_interest = k
            layers[k]['Name'] = bare_name
            layers[k]['EOS'] = int(eos_lines[i].split()[1])
            mesh_words = mesh_lines[i].split()
            layers[k]['Mesh Start'] = int(mesh_words[1])
            layers[k]['Mesh Stop'] = int(mesh_words[2])
            layers[k]['X Start'] = float(mesh_words[3]) * 1e4  # convert centimeters to microns
            layers[k]['X Stop'] = float(mesh_words[4]) * 1e4  # convert centimeters to microns

        return layers, material_of_interest, shock_material_of_interest


class ShockVelocity:
    """Computes and stores the Shock Time and Shock Velocity using Rankine–Hugoniot conditions.

    Note:
        Assumes the Rankine–Hugoniot conditions and in general should not be used for ramp compression simulations.
        Due to Hyades Zone / Mesh indexing, requires the type of indexing used for Particle Velocity.

    Attributes:
            filename (string): Name used when initialized
            dir_name (string): All the preceding directories in the filename
            run_name (string): Only the name of the .inf file (no extension)
            index_mode (string): Input indexing mode used on Particle Velocity
            time (numpy array): Shock time in nanoseconds
            Us (numpy array): Shock velocity, in kilometers per second, at corresponding time
            shock_index (list): Zone index of the computed shock front, one per time
            window_start (list): Starting index of the window where shock front, one per time
            window_stop (list): Ending index of the window where the shock front, one per time

    """
    def __init__(self, filename, mode='Cubic'):
        """Computes and stores the shock velocity profile

        Args:
            filename (string): Name of the .inf
            mode (string): Type of indexing used on particle velocity

        """
        self.filename = filename
        if os.path.isdir(filename):
            self.dir_name = filename
        else:
            self.dir_name = os.path.dirname(filename)
        self.run_name = os.path.splitext(os.path.basename(filename))[0]

        self.index_mode = mode

        self.shock_moi = HyadesOutput(os.path.join(self.dir_name, self.run_name), 'U').shock_moi
        self.time_into_moi = None
        self.time_out_of_moi = None

        time, Us, window_start, window_stop, shock_index = self.calculate_shock_velocity(self.filename, self.index_mode)
        self.time = time
        self.Us = Us
        self.window_start = window_start
        self.window_stop = window_stop
        self.shock_index = shock_index

    def calculate_shock_velocity(self, filename, mode):
        """Compute the Shock Velocity of a simulation using Rankine–Hugoniot conditions.

        Find the shock velocity in a Hyades simulation using the Rankin-Hugoniot equation, Us = P / (Rho_0 * Up)
        which in the code looks like shock_velocity = pressure / (ambient_density * particle_velocity).
        We find the edge of the shock in several steps
        for each time t
            1. Find the zone with the largest Lagrangian Position with a pressure > threshold
            2. Create a window around the zone found in step 1
            3. Find the zone with the highest pressure in the window found in step 2. This is the peak_zone.
            4. Get Pressure at (t, peak_zone), Particle Velocity at (t, peak_zone) and Density at (t_0, peak_zone)
            Shock Velocity = Pressure[t, peak_zone] / (Particle_Velocity[t, peak_zone] * Density[t_0, peak_zone]
        Shock Velocity is not considered meaningful after reaching the free surface, so the calculation stops if the
        window boundaries get very close to the right-hand free surface.

        Note:
            Requires mode due to Mesh / Zone indexing of particle velocity.
            Shock and window indices can be used to plot the position of the shock front.

        Args:
            filename (string): Name of .inf
            mode (string): Indexing method for Particle Velocity - one of Left, Right, Avg

        Returns:
            time (numpy array): Times of the shock velocity in nanoseconds
            shock_velocity (numpy array): Velocity at time t in kilometers per second
            WINDOW_START (list): First index at time t where shock front was searched for
            WINDOW_STOP (list): Last index at time t where shock front was searched for
            SHOCK_INDEX (list): Index of shock front at time t

        # FIXME:
            - My attempt at finding the time in and time out of the shocked material is flawed.
            The strict equality overlooks cases where the shock front jumps several zones and misses the boundary.
            I need to enforce the shock front can only stay still or move to the right, and then use < and >

        """
        hyades_pres = HyadesOutput(filename, 'Pres')
        hyades_rho = HyadesOutput(filename, 'Rho')
        hyades_Up = HyadesOutput(filename, 'U')

        min_index = 8  # only look for a shock front after min_index time steps have occurred
        max_index = len(hyades_pres.time)
        min_pressure = 10  # GPa
        window_size = 10  # check for a shock window_size zones before the leading edge

        pressure = []
        density = []
        particle_velocity = []

        WINDOW_START = []
        WINDOW_STOP = []
        SHOCK_INDEX = []

        for t in range(min_index, max_index):
            try:
                # leading edge is the furthest-right zone index where the pressure is greater than min_pressure
                leading_edge = max(np.where(hyades_pres.output[t, :] > min_pressure)[0])
            except ValueError:
                print(f'Time: {t, hyades_pres.time[t]}, Max Pressure: {hyades_pres.output[t, :].max()}')
                fig, ax = plt.subplots()
                ax.plot(hyades_pres.x[0, :], hyades_pres.output[t, :])
                ax.set_title(f'Error Graph at {hyades_pres.time[t]:.2f} ns')
                ax.set(xlabel='Lagrangian Distance (um)', ylabel='Pressure (GPa)')
                plt.show()
                raise Exception(f'At {hyades_pres.time[t]} ns could not find a pressure greater than {min_pressure} GPa,'
                                f'which caused the shock velocity calculation to crash.')

            '''shock_index is where we consider the shock front to be. See function description for details.'''
            window_start = leading_edge - window_size
            if window_start < 0:
                window_start = 0
            window_stop = leading_edge
            pressure_window = hyades_pres.output[t, window_start:window_stop]
            shock_index = window_start + np.argmax(pressure_window)  # Shock index is the

            WINDOW_START.append(window_start)
            WINDOW_STOP.append(window_stop)
            SHOCK_INDEX.append(shock_index)

            pressure.append(hyades_pres.output[t, shock_index])
            density.append(hyades_rho.output[0, shock_index])

            left = hyades_Up.output[t, shock_index]
            right = hyades_Up.output[t, shock_index + 1]
            if mode.lower() == 'ucm':
                '''Attempt to load UCM, which is the Zone-indexed particle velocity output by Hyades'''
                try:
                    ucm = HyadesOutput(filename, 'UCM')
                    Up = ucm.output[t, shock_index]
                except KeyError as e:
                    run_name = os.path.splitext(os.path.basename(filename))[0]
                    print(f'UCM was specified, but was not found in {run_name}.cdf\n'
                          f'Check if ucm is in pparray line in {run_name}.inf')
                    raise e
            elif (mode.lower() == 'left') or (mode == 'L'):
                Up = left
            elif (mode.lower() == 'right') or (mode == 'R'):
                Up = right
            elif (mode.lower() == 'average') or (mode.lower() == 'avg'):
                Up = (left + right) / 2
            elif mode.lower() == 'cubic':  # Interpolate Particle Velocity with Cubic Spline
                x = hyades_Up.x[t, :]
                y = hyades_Up.output[t, :]
                cubic_spline = CubicSpline(x, y)
                zone_x = hyades_pres.x[t, shock_index]
                Up = cubic_spline(zone_x)
            else:
                raise ValueError(f'Shock Velocity Interpolation Mode {mode!r} not recognized. '
                                 f'Use one of Left, Right, Average, Cubic, Ucm')
            particle_velocity.append(Up)

            '''Attempting to find the time the shock enters and exits the shock material of interest.'''
            if self.shock_moi:  # Only True if inf has a shock material of interest specified
                if self.time_into_moi is None:  # only consider reassigning time_into_moi if it is None
                    if shock_index == hyades_Up.layers[hyades_Up.shock_moi]['Mesh Start']:
                        self.time_into_moi = hyades_Up.time[t]
                if self.time_out_of_moi is None:  # only consider reassigning time_out_of_moi if it is None
                    if shock_index == hyades_Up.layers[hyades_Up.shock_moi]['Mesh Stop']:
                        self.time_out_of_moi = hyades_Up.time[t]

            if abs(len(hyades_pres.x[0, :]) - window_stop) <= 2:  # If the shock index is too close to the free surface, stop
                break

        shock_velocity = np.array(pressure) / (np.array(density) * np.array(particle_velocity))
        time = hyades_pres.time[min_index:t + 1]

        return time, shock_velocity, WINDOW_START, WINDOW_STOP, SHOCK_INDEX
