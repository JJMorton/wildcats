import sys
array_id=int(sys.argv[1])
print(f'Running {sys.argv[0]} with {array_id=}')

import numpy as np
import pandas as pd
from sim.model import simple_sim
from sim.sum_stats import simple_sum

base_seed = 200
num_params = 16
num_stats = 102


def simulator(params, seed):
	"""
	Takes the 16 parameters as input, returns the summary statistics as a list
	"""
	if len(params) != num_params:
		raise IndexError(f'Incorrect number of parameters passed to simulator (want {num_params}, got {len(params)})')

	length=int(10e6)
	recombination_rate=1.8e-8
	mutation_rate=6e-8

	print("Starting simulation...")
	print(f'{params=}')
	print(f'{seed=}')
	data = simple_sim(*params, length, recombination_rate, mutation_rate, seed=seed)
	
	print("Calculating summary statistics...")
	summary_stats = simple_sum(data)

	if len(summary_stats) != num_stats:
		raise IndexError(f'Simulator outputted incorrect number of summary statistics (want {num_stats}, got {len(summary_stats)})')
	
	print("Simulation complete")
	return list(summary_stats.values())

def write_stats(stats, seed):
	np.savetxt(f'../output/stats/stats_{array_id}.csv', [stats + [seed]], delimiter=',')


seed = base_seed + array_id
params_df = pd.read_feather("../data/simulate_params.feather")

# Write empty stats now, in case job times out and script doesn't complete
stats = [np.nan] * num_stats
write_stats(stats, seed)

try:
	if array_id < 0 or array_id >= params_df.values.shape[0]:
		raise IndexError(f'Array ID not within range defined by number of rows in parameters table ({params_df.values.shape[0]})')
	theta = params_df.values[array_id]
	stats = simulator(theta, seed)
	write_stats(stats, seed)
except:
	print("Failed to calculate summary statistics, using list of np.nan")

