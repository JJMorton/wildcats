# Check the inference/config.py configuration file, make sure everything is ready for simulations and inference

import inference.config as config
from inference.priors import join_priors
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
    print('1. Checking directory structure')

    for dirpath in np.unique([config.simulation_output_dir, path.dirname(config.posterior_plot_file), path.dirname(config.parameters_file), path.dirname(config.stats_file)]):
        if not path.isdir(dirpath):
            print(f'Directory "{dirpath}" doesn\'t exist, creating...')
            os.makedirs(dirpath)
        else:
            print(f'Directory "{dirpath}" exists')

    print('======================================')
    print(f'2. Testing proposal distribution in "{config.proposal_pickle_file}"...')
    try:
        with open(config.proposal_pickle_file, 'rb') as f:
            proposal = pickle.load(f)
    except:
        if input(f'Proposal file "{config.proposal_pickle_file}" does not exist, would you like to dump the prior to it? [y/n] ') == 'y':
            proposal = join_priors()
            with open(config.proposal_pickle_file, 'wb') as f:
                pickle.dump(proposal, f)
        else:
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
    print('3. Checking the csv files')
    for file in (config.parameters_file, config.stats_file):
        if path.exists(file):
            try:
                data = pd.read_csv(file, index_col='index')
            except:
                print(f'"{file}" is an invalid format')
                exit(1)
            print(f'WARNING: "{file}" is of shape {data.shape}, ensure this file has valid contents if you\'re going to use it')
        else:
            print(f'WARNING: "{file}" doesn\'t exist, we are okay to create it')

    try:
        observations = pd.read_csv(config.observation_file).values[0]
    except:
        print(f'Could not read observation from "{config.observation_file}", check that the file exists')
        exit(1)
    if len(observations) != stats_length:
        print(f'Observation is length {len(observations)}, should be {stats_length}')
        exit(1)
    else:
        print('Observation file is valid')
    
    print('======================================')
    print('4. Checking output files')
    for file in (config.posterior_pickle_file, config.posterior_plot_file):
        if path.exists(file):
            print(f'WARNING: "{file}" already exists, if you run inference again it will be overwritten.')
    
    if len(os.listdir(config.simulation_output_dir)) > 0:
        print(f'WARNING: "{config.simulation_output_dir}" is not empty, any files named stats_*.csv may be overwritten')

    if not path.exists(config.parameters_file):
        print('======================================')
        print('4. Taking samples from proposal')
        
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
