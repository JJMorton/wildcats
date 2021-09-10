# Check the inference/config.py configuration file, make sure everything is ready for simulations and inference

import inference.config as config
import pickle
from torch import tensor
import os
import os.path as path
import numpy as np
import inference.priors
from sbi.utils.user_input_checks import process_prior
import pandas as pd

def main():

    sample_length = 16
    sample_batch_shape = (config.num_simulations,)
    stats_length = 102

    print('Checking inference/config.py, make sure this script completes and outputs "OK"')
    
    print('======================================')
    print('1. Checking directory structure')

    dirs = [
        config.simulation_output_dir,
        path.dirname(config.posterior_plot_file),
        path.dirname(config.parameters_file),
        path.dirname(config.stats_file),
        "../output/slurm"
    ]
    for dirpath in np.unique(dirs):
        if not path.isdir(dirpath):
            print(f'Directory "{dirpath}" doesn\'t exist, creating...')
            os.makedirs(dirpath)
        else:
            print(f'Directory "{dirpath}" exists')

    print('======================================')
    print('2. Testing prior and proposal distributions...')

    if path.exists(config.prior_pickle_file):
        with open(config.prior_pickle_file, 'rb') as f:
            prior = pickle.load(f)
    else:
        print(f'Prior file "{config.prior_pickle_file}" does not exist, creating...')
        prior = process_prior(inference.priors.join_priors())[0]
        with open(config.prior_pickle_file, 'wb') as f:
            pickle.dump(prior, f)

    if path.exists(config.proposal_pickle_file):
        with open(config.proposal_pickle_file, 'rb') as f:
            proposal = pickle.load(f)
    else:
        print(f'Proposal file "{config.proposal_pickle_file}" does not exist.')
        exit(1)

    # Make sure the samples and probabilities are the correct shape
    for dist, name in zip((prior, proposal), ("prior", "proposal")):
        samples = dist.sample(sample_batch_shape)
        if samples.shape != (*sample_batch_shape, sample_length):
            print(f'Samples from {name} are of an incorrect shape (wanted {(*sample_batch_shape, sample_length)}, got {samples.shape})')
            exit(1)
        else:
            print(f'Samples from {name} are of the correct shape')
        probs = dist.log_prob(samples)
        if probs.shape != sample_batch_shape:
            print(f'Probabilities from {name} are of an incorrect shape (wanted {sample_batch_shape}, got {probs.shape})')
            exit(1)
        else:
            print(f'Probabilities from {name} are of the correct shape')

    within_prior = prior.log_prob(proposal.sample((10_000,))).isfinite()
    within_prior_count = within_prior[within_prior].shape[0]
    if within_prior_count == 10_000:
        print('All proposal samples are within prior.')
    else:
        print(f'WARNING: {within_prior_count} of 10,000 proposal samples are within the prior')

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
