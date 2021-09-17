import inference.config as config
import pandas as pd
import torch
from torch import tensor
import pickle
import os.path as path

# Various functions for reading and saving data

def strip_filename(filepath):
    """returns just the filename, without its extension"""
    return path.splitext(path.basename(filepath))[0]

def get_all_theta():
    params_df = pd.read_csv(config.parameters_file, index_col="index")
    return tensor(params_df.values, dtype=torch.float32)

def get_all_x():
    stats_df = pd.read_csv(config.stats_file, index_col="index").drop(columns=["seed"])
    return tensor(stats_df.values, dtype=torch.float32)

def get_theta_x():
    # Read the csv files and create torch tensors for use with sbi
    
    stats_df = pd.read_csv(config.stats_file, index_col="index").drop(columns=["seed"])
    # Only take the parameter sets that there are summary statistics for in the table above
    params_df = pd.read_csv(config.parameters_file, index_col="index")
    params_df = params_df[params_df.index.isin(stats_df.index)]

    # Get the parameters and outputs into torch tensors
    theta = tensor(params_df.values, dtype=torch.float32)
    x = tensor(stats_df.values, dtype=torch.float32)
    
    return theta, x

def remove_outside_prior(prior, theta, x):
    valid_samples = prior.log_prob(theta).isfinite()
    print(f'{valid_samples[valid_samples].shape[0]} of {theta.shape[0]} samples are within prior')
    return theta[valid_samples], x[valid_samples]

def get_observation():
    return tensor(pd.read_csv(config.observation_file).values, dtype=torch.float32)[0]

def get_prior():
    with open(config.prior_pickle_file, 'rb') as f:
        prior = pickle.load(f)
    return prior

def get_proposal():
    with open(config.proposal_pickle_file, 'rb') as f:
        proposal = pickle.load(f)
    return proposal

def save_posterior(posterior):
    with open(config.posterior_pickle_file, 'wb') as f:
        pickle.dump(posterior, f)
        print(f'Dumped posterior to "{config.posterior_pickle_file}"')
