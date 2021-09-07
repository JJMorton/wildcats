import sys
import inference.config as config
from os.path import join as joinpath

import numpy as np
import pandas as pd
from sim.model import simple_sim
from sim.sum_stats import simple_sum

def simulator(params, seed):
    """
    Takes the 16 parameters as input, returns the summary statistics as a list
    """

    length=int(10e6)
    recombination_rate=1.8e-8
    mutation_rate=6e-8

    print("Starting simulation...")
    print(f'{params=}')
    print(f'{seed=}')
    data = simple_sim(*params, length, recombination_rate, mutation_rate, seed=seed)
    
    print("Calculating summary statistics...")
    summary_stats = simple_sum(data)
    
    print("Simulation complete")
    return list(summary_stats.values())

def write_stats(stats, seed, filename):
    np.savetxt(joinpath(config.simulation_output_dir, filename), [stats + [seed]], delimiter=',')

def main():
    array_id=int(sys.argv[1])
    print(f'Running {sys.argv[0]} with {array_id=}, using parameters from "{config.parameters_file}"')

    base_seed = 200
    num_params = 16
    num_stats = 102

    seed = base_seed + array_id
    params_df = pd.read_csv(config.parameters_file, index_col="index")
    output_filename = f'stats_{array_id}.csv'

    # Write empty stats now, in case job times out and script doesn't complete
    stats = [np.nan] * num_stats
    write_stats(stats, seed, output_filename)

    try:
        if array_id < 0 or array_id >= params_df.values.shape[0]:
            raise IndexError(f'Array ID not within range defined by number of rows in parameters table ({params_df.values.shape[0]})')
        theta = params_df.values[array_id]
        if len(theta) != num_params:
            raise IndexError(f'Incorrect number of parameters passed to simulator (want {num_params}, got {len(theta)})')
        stats = simulator(theta, seed)
        if len(stats) != num_stats:
            raise IndexError(f'Simulator output incorrect number of summary statistics (want {num_stats}, got {len(stats)})')
        write_stats(stats, seed, output_filename)
    except:
        print("Failed to calculate summary statistics, saved list of np.nan")


if __name__ == "__main__":
    main()
