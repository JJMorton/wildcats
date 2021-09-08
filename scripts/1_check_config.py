# Check the inference/config.py configuration file, make sure everything is ready for simulations and inference

import inference.config as config
import pickle
from torch import tensor
import os
import os.path as path
import numpy as np
import inference.priors
import pandas as pd

def main():

    sample_length = 16
    sample_batch_shape = (config.num_simulations,)
    stats_length = 102

    print('Checking inference/config.py, make sure this script completes and outputs "OK"')

    print('======================================')
    print(f'1. Testing proposal distribution in "{config.proposal_pickle_file}"...')
    try:
        with open(config.proposal_pickle_file, 'rb') as f:
            proposal = pickle.load(f)
    except:
        print(f'Failed to open proposal file')
        exit(1)
    # Make sure the samples and probabilities are the correct shape
    samples = proposal.sample(sample_batch_shape)
    if samples.shape != (*sample_batch_shape, sample_length):
        print(f'Samples are of an incorrect shape (wanted {(*sample_batch_shape, sample_length)}, got {samples.shape}), check the proposal')
        exit(1)
    else:
        print('Samples are of the correct shape')
    probs = proposal.log_prob(samples)
    if probs.shape != sample_batch_shape:
        print(f'Probabilities are of an incorrect shape (wanted {sample_batch_shape}, got {probs.shape}), check the proposal')
        exit(1)
    else:
        print('Probabilities are of the correct shape')

    print('======================================')
    print('2. Checking the csv files')
    for file in (config.parameters_file, config.stats_file):
        directory = path.dirname(file)
        if path.isdir(directory):
            print(f'Directory "{directory}" exists')
        else:
            print(f'Directory "{directory}" doesn\'t exist, creating...')
            os.makedirs(directory)

        if path.exists(file):
            print(f'"{file}" already exists. IMPORTANT: Ensure this file has valid contents if you\'re going to use it')
        else:
            print(f'"{file}" doesn\'t exist, we are okay to create it')

    try:
        observations = np.loadtxt(config.observation_file, delimiter=',')
    except:
        print(f'Could not read observation from "{config.observation_file}", check that the file exists')
        exit(1)
    if len(observations) != stats_length:
        print(f'Observation is length {len(observations)}, should be {stats_length}')
        exit(1)
    else:
        print('Observation file is valid')
    
    print('======================================')
    print('2. Checking the simulation output directory')

    if not path.isdir(config.simulation_output_dir):
        print(f'Directory "{config.simulation_output_dir}" doesn\'t exist, creating...')
        os.makedirs(config.simulation_output_dir)
    else:
        print(f'Directory "{config.simulation_output_dir}" exists')

    if not path.exists(config.parameters_file):
        print('======================================')
        print('3. Taking samples from proposal')
        
        with open(config.proposal_pickle_file, 'rb') as f:
            proposal = pickle.load(f)

        samples = proposal.sample((config.num_simulations,))
        param_names = inference.priors.get_param_names()
        pd.DataFrame(np.array(samples), columns=param_names).to_csv(config.parameters_file, index=True, index_label="index")
        print(f'Saved proposal samples to "{config.parameters_file}"')

    print('======================================')
    print('OK')


if __name__ == "__main__":
    main()
