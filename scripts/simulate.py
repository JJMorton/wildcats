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
    return list(summary_stats.keys()), list(summary_stats.values())

def write_stats(index, stats, cols, seed, filename):
    filepath = joinpath(config.simulation_output_dir, filename)
    pd.DataFrame([[index] + stats + [seed]], columns=['index'] + cols + ['seed']).to_csv(filepath, index=False)

def main():
    array_id=int(sys.argv[1])
    print(f'Running {sys.argv[0]} with {array_id=}, using parameters from "{config.parameters_file}"')

    base_seed = 200
    num_params = 16
    num_stats = 102

    seed = base_seed + array_id
    params_df = pd.read_csv(config.parameters_file, index_col="index")
    output_filename = f'stats_{array_id}.csv'

    try:
        if array_id < 0 or array_id >= params_df.values.shape[0]:
            raise IndexError(f'Array ID not within range defined by number of rows in parameters table ({params_df.values.shape[0]})')
        theta = params_df.values[array_id]
        if len(theta) != num_params:
            raise IndexError(f'Incorrect number of parameters passed to simulator (want {num_params}, got {len(theta)})')
        names, stats = simulator(theta, seed)
        if len(stats) != num_stats:
            raise IndexError(f'Simulator output incorrect number of summary statistics (want {num_stats}, got {len(stats)})')
        write_stats(array_id, stats, names, seed, output_filename)
    except Exception as e:
        print(str(e))
        print("Failed to calculate summary statistics")


if __name__ == "__main__":
    main()
