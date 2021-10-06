"""Functions to run Hyades, convert the .otf to .cdf, and organize the output files into folders.

Todo:
    * Confirm this works with the inf_GUI.py
    * Confirm the logging is working
"""
import os
import time
import shutil
import logging
import subprocess

from tools.excel_writer import write_excel


def run_hyades(inf_name):
    """Runs a single Hyades simulation.

    Args:
        inf_name (string): Name of the .inf

    Returns:
        terminal output, terminal error message, terminal response code

    """
    cmd = f'hyades {inf_name}'
    sp = subprocess.Popen(cmd,
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
    response_code = sp.wait()
    out, err = sp.communicate()

    return out, err, response_code


def otf2cdf(otf_name):
    """Runs the PPF2NCDF command to convert Hyades output (.otf) to a netcdf (.cdf) file

    Args:
        otf_name (string): Name of the .otf (should match name of .inf)

    Returns:
        terminal output, terminal error message, terminal response code

    """
    cmd = f'PPF2NCDF {os.path.splitext(otf_name)[0]}'
    print(cmd)
    sp = subprocess.Popen(cmd,
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
    response_code = sp.wait()
    out, err = sp.communicate()

    return out, err, response_code


def batch_run_hyades(inf_dir, out_dir, excel_variables=[]):
    """Runs Hyades simulations of many .inf files and packages each output into its own folder.

    Args:
        inf_dir (string): Name of the directory containing .inf files
        out_dir (string): Destination directory where all the data will end up
        excel_variables (list, optional): List of abbreviated variable names to copy to excel file

    Returns:
        None

    """
    inf_files = [f for f in os.listdir(inf_dir) if f.endswith('.inf')]
    if len(inf_files) == 0:  # if there are no inf files in the inf_directory
        raise ValueError(f'Did not find any .inf files in {inf_dir}')

    # Set up a logging file
    filename = 'hyades.log'
    log_format = '%(asctime)s %(levelname)s:%(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename=filename, format=log_format, datefmt=date_format, level=logging.DEBUG)

    for inf in inf_files:
        print(f'Starting Hyades {inf}')
        abs_path = os.path.join(inf_dir, inf)
        t0 = time.time()
        os.system(f'hyades {abs_path}')
        # Run Hyades and post processor
        # out, err, rc = run_hyades(abs_path)
        # if err:
        #     raise Exception(f'Error from terminal while running "Hyades {abs_path}":\n{err}')
        t1 = time.time()
        log_note = f'Completed Hyades simulation of {inf} in {t1-t0:.2f} seconds.'
        out, err, rc = otf2cdf(abs_path)
        if err:
            raise Exception(f'Error from terminal while running "PPF2NCDF {abs_path}":\n{err}')
        log_note += ' Completed PPF2NCDF.'
        # Optionally save .cdf as a human-readable excel file
        if excel_variables:
            excel_filename = os.path.join(abs_path, os.path.splitext(inf)[0])
            write_excel(abs_path, excel_filename, excel_variables)
            log_note += f' Saved {", ".join(excel_variables)} to excel file.'
        logging.info(log_note)

        # Create new directory in out_dir
        basename = os.path.splitext(inf)[0]
        new_dir = os.path.join(out_dir, basename)
        os.mkdir(new_dir)
        print(f'Made {new_dir}')

        # Move all files with the same name as the .inf to the new directory
        for f in os.listdir(inf_dir):
            if os.path.splitext(f)[0] == basename:
                source = os.path.join(inf_dir, f)
                destination = os.path.join(new_dir, f)
                shutil.move(source, destination)
                print(f'moved {destination}')

    return None
