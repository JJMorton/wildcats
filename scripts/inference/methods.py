import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import torch
import torch.distributions
from torch import tensor
from sbi.inference import SNLE, SNPE, prepare_for_sbi
from sbi.utils.user_input_checks import process_prior
from sbi import utils as utils
from sbi import analysis as analysis
import scipy.stats
import scipy.stats.mstats

from inference import priors
from inference.analysis import kl_divergence

# class Sampler:
#     def __init__(self, distribution, is_normalised):
#         self._is_normalised = is_normalised
#         self.distribution = distribution
    
#     def __str__(self):
#         return f'Sampler to sample from {"" if self._is_normalised else "un"}normalised distribution:\n{str(self.distribution)}'
        
#     def sample(self, *args, **kwargs):
#         """
#         Samples from the distribution and unnormalises the samples if necessary
#         """
#         samples = self.distribution.sample(*args, **kwargs)
#         if self._is_normalised:
#             samples = priors.unnormalise_samples(samples)
#         return samples
    
#     def log_prob(self, samples, *args, **kwargs):
#         """
#         Evaluates the distribution's log_prob with the samples, first normalising them if necessary
#         """
#         if self._is_normalised:
#             return self.distribution.log_prob(priors.normalise_samples(samples), *args, **kwargs)
#         else:
#             return self.distribution.log_prob(samples, *args, **kwargs)


def load_posterior(filename):
    try:
        with open(filename, "rb") as f:
            posterior = pickle.load(f)
            print(f'Loaded posterior from "{filename}"')
            print(posterior)
            return posterior
    except Exception as e:
        logging.error(e)
        logging.error(f'Failed to load posterior from "{filename}"')
        return None

def dump_posterior(posterior, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(posterior, f)
            print(f'Dumped posterior to "{filename}"')
    except Exception as e:
        print(e)
        logging.error(f'Failed to dump posterior to "{filename}"')

def method_npe(
    theta,
    x,
    density_estimator='maf',
    training_batch_size=10_000,
    dump_to_file=None,
    posterior_args={},
    training_args={}
):
    # Make sure we don't modify the `theta` provided to us
    theta = theta.clone().detach()
    prior = priors.join_priors()
    num_simulations = theta.shape[0]
    training_batch_size = min(training_batch_size, num_simulations)
    
    inference = SNPE(prior=process_prior(prior)[0], density_estimator=density_estimator)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(show_train_summary=True, training_batch_size=training_batch_size, **training_args)
    posterior = inference.build_posterior(density_estimator, sample_with_mcmc=False, **posterior_args)
    
    if dump_to_file is not None:
        dump_posterior(posterior, dump_to_file)
    
    return posterior

def method_snpe(
    theta,
    x,
    x_o,
    num_rounds=2,
    density_estimator='maf',
    training_batch_size=50,
    dump_to_file=None,
    calculate_kl=False,
    posterior_args={},
    training_args={}
):
    # Make sure we don't modify the `theta` provided to us
    theta = theta.clone().detach()
    
    num_simulations = theta.shape[0]
    num_simulations_per_round = math.floor(num_simulations / num_rounds)
    training_batch_size = min(training_batch_size, num_simulations_per_round)
    def_training_args = {
        'training_batch_size': training_batch_size,
        'retrain_from_scratch_each_round': False,
        'discard_prior_samples': False,
        'use_combined_loss': False
    }
    def_training_args.update(training_args)
    
    posteriors = []
    proposal = prior
    inference = SNPE(prior=process_prior(prior)[0], density_estimator=density_estimator)
    
    for r in range(num_rounds):
        print(f'Starting round {r + 1}/{num_rounds}')
        theta_round = theta[r * num_simulations_per_round : (r + 1) * num_simulations_per_round]
        x_round = x[r * num_simulations_per_round : (r + 1) * num_simulations_per_round]
        try:
            density_estimator = inference.append_simulations(
                theta_round, x_round, proposal=proposal
            ).train(**training_args)
            posterior = inference.build_posterior(density_estimator, sample_with_mcmc=False, **posterior_args)
        except Exception as err:
            logging.error(err)
            print(f'Round {r + 1} failed, returning posterior of previous round')
            posterior = posteriors[-1] if len(posteriors) > 0 else None
            break
            
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)
        kl = kl_divergence(prior, posterior, x_o, base=2)
        print(f'KL divergence: {kl}')
        dump_posterior(posterior, '.'.join(dump_to_file.split('.')[:-1]) + str(r + 1) + '.pkl')
    
    if dump_to_file is not None:
        dump_posterior(posterior, dump_to_file)
    
    return posterior

def method_nle(
    theta,
    x,
    density_estimator='maf',
    training_batch_size=10_000,
    dump_to_file=None,
    mcmc_method='slice_np',
    posterior_args={},
    training_args={},
    mcmc_parameters={}
):
    # Make sure we don't modify the `theta` provided to us
    theta = theta.clone().detach()
    prior = priors.join_priors()
    num_simulations = theta.shape[0]
    training_batch_size = min(training_batch_size, num_simulations)
    
    inference = SNLE(prior=process_prior(prior)[0], density_estimator=density_estimator)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(show_train_summary=True, training_batch_size=training_batch_size, **training_args)
    posterior = inference.build_posterior(density_estimator, mcmc_method=mcmc_method, mcmc_parameters=mcmc_parameters, **posterior_args)
    
    if dump_to_file is not None:
        dump_posterior(posterior, dump_to_file)
    
    return posterior
    