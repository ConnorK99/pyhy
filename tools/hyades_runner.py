"""Functions to run Hyades, convert the .otf to .cdf, and organize the output files into folders."""
import os
import time
import shutil
import logging
import subprocess

from tools.excel_writer import write_excel


def run_hyades(inf_name, quiet=False):
    """Runs a single Hyades simulation.

    Args:
        inf_name (string): Name of the .inf
        quiet (bool, optional): Toggle to save the terminal output to a text file instead of printing on screen.
                                This text file is automatically deleted.

    Returns:
        log_string (string): Status and details of Hyades simulation

    """
    if quiet:
        txt_file = os.path.splitext(inf_name)[0] + '_hyades_terminal.txt'
        command = f'hyades {inf_name} > {txt_file}'
    else:
        command = f'hyades {inf_name}'

    t0 = time.time()
    os.system(command)
    t1 = time.time()

    if quiet:
        if os.path.exists(txt_file):  # Delete the terminal output if it exists
            os.remove(txt_file)

    file_extensions = ('.otf', '.ppf', '.tmf')
    run_name = os.path.basename(os.path.splitext(inf_name)[0])
    found_all = all([run_name + ext in os.listdir() for ext in file_extensions])
    if found_all:
        log_string = f'Completed Hyades simulation of {os.path.basename(inf_name)} in {t1 - t0:.2f} seconds.'
    else:
        log_string = f'Failed to run Hyades simulation of {os.path.basename(inf_name)}.'

    return log_string


def otf2cdf(otf_name, quiet=False):
    """Runs the PPF2NCDF command to convert Hyades output (.otf) to a netcdf (.cdf) file

    Args:
        otf_name (string): Name of the .otf (should match name of .inf)
        quiet (bool, optional): Toggle to save the terminal output to a text file instead of printing on screen

    Returns:
        log_string (string): status of the PPF2NCDF command

    """
    if quiet:
        txt_file = os.path.splitext(otf_name)[0] + '_PPF2NCDF_terminal.txt'
        command = f'PPF2NCDF {os.path.splitext(otf_name)[0]} > {txt_file}'
    else:
        command = f'PPF2NCDF {os.path.splitext(otf_name)[0]}'
    os.system(command)

    if quiet:
        if os.path.exists(txt_file):  # Delete the terminal output if it exists
            os.remove(txt_file)
    '''On windows, Hyades creates a file called SCRACTHX1 that just has a tab and a "1" in it.
    I have no idea why Hyades does this but leaving the file in there can crash the optimizer.'''
    if os.path.exists('SCRATCHX1'):
        os.remove('SCRATCHX1')

    run_name = os.path.basename(os.path.splitext(otf_name)[0])
    found = run_name + '.cdf' in os.listdir()
    if found:
        log_string = 'Completed PPF2NCDF.'
    else:
        log_string = 'Failed PPF2NCDF.'

    return log_string


def batch_run_hyades(inf_dir, out_dir, excel_variables=[], quiet=False):
    """Runs Hyades simulations of many .inf files and packages each output into its own folder.

    Note:
        According to page 13 of the Hyades User Guide, best practice is to change to the directory with the .inf
        and run Hyades from there. This function moves into the directory with the .inf, runs Hyades and the post-
        processor, then changes back to the original directory.

    Args:
        inf_dir (string): Name of the directory containing .inf files
        out_dir (string): Destination directory where all the data will end up
        excel_variables (list, optional): List of abbreviated variable names to copy to excel file
        quiet (bool, optional): Toggle to hide the terminal output during simulation

    Returns:
        None

    """
    inf_files = sorted([f for f in os.listdir(inf_dir) if f.endswith('.inf')])
    if len(inf_files) == 0:  # if there are no inf files in the inf_directory
        raise ValueError(f'Did not find any .inf files in {inf_dir}')

    if inf_dir.startswith('./') or inf_dir.startswith('.\\'):  # Hyades does not work with ./ or .\ prepended
        inf_dir = inf_dir[2:]

    # Set up a logging file
    filename = 'hyades.log'
    log_format = '%(asctime)s %(levelname)s:%(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename=filename, format=log_format, datefmt=date_format, level=logging.DEBUG)

    old_directory = os.getcwd()
    for inf in inf_files:
        os.chdir(inf_dir)  # Change to the directory where the file is
        try:
            log_note = run_hyades(inf, quiet=quiet)  # Run Hyades
            log_note += ' ' + otf2cdf(inf, quiet=quiet)  # Run PPF2NCDF to create .cdf file and add note to log
        finally:
            os.chdir(old_directory)  # Change back to the original directory
        # Optionally convert .cdf as a human-readable Excel file
        if excel_variables:
            absolute_path = os.path.join(inf_dir, inf)
            excel_filename = os.path.join(os.path.splitext(absolute_path)[0])
            write_excel(absolute_path, excel_filename, excel_variables)
            log_note += f' Saved {", ".join(excel_variables)} to excel file.'
        logging.info(log_note)

        # Create new directory in out_dir
        basename = os.path.splitext(inf)[0]
        new_dir = os.path.join(out_dir, basename)
        os.mkdir(new_dir)
        # Move all files with the same name as the .inf to the new directory
        for f in os.listdir(inf_dir):
            if os.path.splitext(f)[0] == basename:
                source = os.path.join(inf_dir, f)
                destination = os.path.join(new_dir, f)
                shutil.move(source, destination)

    return None
