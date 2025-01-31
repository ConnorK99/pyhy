"""An attempt to run Hyades simulations in parallel. Does **not** neatly format the output like hyades_runner.py"""
import os
import time
import multiprocessing


def run_hyades(inf):
    """A simple function to run Hyades in parallel for several inf files"""
    command = f'hyades {inf} > {inf}_terminal_output.txt'
    with open('parallel_log.txt', 'a') as f:
        f.write(f'Starting {command}\n')
    start = time.time()
    os.system(command)
    stop = time.time()
    with open('parallel_log.txt', 'a') as f:
        f.write(f'Finished {command} in {stop - start:.4f} seconds\n')


if __name__ == '__main__':
    path = '../data/inf'
    inf_files = sorted([f for f in os.listdir(path) if f.endswith('.inf')])
    with open('parallel_log.txt', 'w') as f:
        f.write(f'Logging for parallel_log.py\n')
        f.write(f'Running {len(inf_files)} Hyades simulations in parallel.\n')
        f.write(f'inf files are: {", ".join(inf_files)}\n')
    old_dir = os.getcwd()
    os.chdir(path)
    # Begin Parallel processing
    num_processes = len(inf_files)
    start_time = time.time()
    process_pool = multiprocessing.Pool(processes=num_processes)
    inf_files = [os.path.join(path, inf) for inf in inf_files]
    process_pool.map(run_hyades, inf_files)
    process_pool.close()
    end_time = time.time()
    # End parallel processing
    os.chdir(old_dir)
    with open('parallel_log.txt', 'a') as f:
        f.write(f'Entire script took {end_time - start_time:.4f} seconds.')
