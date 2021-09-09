# To run this, we need the following:
#  1. the pickled proposal (either the prior or a posterior)
#  2. the parameter sets in csv format
#  3. the corresponding summary statistics in csv format
#     (some may be missing if any simulations failed to run, this script deals with that)
#  4. the csv of the observation we made (a single set of summary statistics)

import inference.config as config
import pandas as pd
from torch import tensor
import torch
import pickle
from sbi.inference import SNPE
from sbi.utils.user_input_checks import process_prior
import os.path as path
import matplotlib.pyplot as plt
from inference.analysis import plot_samples_vs_prior
from inference.priors import join_priors

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

def dump_posterior(posterior, filename):
    with open(filename, 'wb') as f:
        pickle.dump(posterior, f)
        print(f'Dumped posterior to "{filename}"')

def main():
    
    print("Importing data...")
    theta, x = get_theta_x()
    x_o = get_observation()
    proposal = get_proposal()
    prior = get_prior()
    
    print(f'{theta.shape=}')
    print(f'{x.shape=}')
    print(f'{x_o.shape=}')
    print(f'{type(proposal)=}')
    print(f'{type(prior)=}')
    
    print("Running inference...")
    inference = SNPE(prior=prior, density_estimator='maf')
    inference = inference.append_simulations(theta, x, proposal=proposal)
    density_estimator = inference.train(show_train_summary=True, training_batch_size=50)
    posterior = inference.build_posterior(density_estimator, sample_with_mcmc=False)
    posterior.set_default_x(x_o)
    dump_posterior(posterior, config.posterior_pickle_file)
    
    print("Plotting posterior samples...")
    fig, _ = plot_samples_vs_prior(prior, posterior.sample((10_000,)), "posterior")
    plt.savefig(config.posterior_plot_file)

if __name__ == "__main__":
    main()
