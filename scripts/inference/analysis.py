import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
import logging
from sbi import analysis
import scipy.stats
import scipy.stats.mstats
import math

# from torch.distributions.kl import kl_divergence
from .two_nn import twonn_dimension

def kl_divergence(prior, posterior, x, num_samples=2000, base=math.e):
    samples = posterior.sample((num_samples,), x=x, show_progress_bars=False)
#     sample_size = len(posterior.sample(show_progress_bars=False))
#     samples = torch.zeros((num_samples, sample_size))
#     for i in range(num_samples):
#         print(f'Sampling... [{i+1}/{num_samples}]', end='', flush=True)
#         samples[i] = posterior.sample(x=x, show_progress_bars=False)
#         print("\r                                  \r", end='')
#     print(f'Sampled using {num_samples} observations')
    
    def remove_invalid(t):
        return t[~(t.isnan() | t.isinf())]
    
    prob_prior = prior.log_prob(samples) / math.log(base)
    prob_posterior = posterior.log_prob(samples, x=x) / math.log(base)
#     diff = torch.exp(prob_posterior) * (prob_posterior - prob_prior)
    diff = (prob_posterior - prob_prior)
    
    # Remove all the nans and infs
    diff = diff[~(diff.isnan() | diff.isinf())]
    N = len(diff)
    if N < num_samples:
        logging.info(f'Removed samples outside of prior, using {N}/{num_samples} samples')
    
    return float(torch.sum(diff) / N)

def create_params_plot(prior, axsize=4, num_cols=4):
    """
    Creates an empty plot with as many subplots as there are parameters in the prior
    """
    num_plots = len(prior.dists)
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, constrained_layout=True, figsize=(axsize * num_cols, axsize * num_rows))
    for ax, param in zip(axes.flat, prior.params):
        ax.set_title(param)
    return fig, axes.flat

def plot_samples_vs_prior(prior, sample_sets, sample_labels, actual_theta=None, axsize=4, num_cols=4):
    """
    Plot a histogram of each set of samples, together with the prior
    `sample_sets` should be of the shape (number of sets, number of samples, number of parameters), but can exclude dimension 0 if there is only one set.
    `sample_labels` should be of the shape (number of sets) and contains the samples' labels for the legend
    """
    
    # Make sure samples are in correct format
    sample_sets = np.array(sample_sets)
    sample_labels = np.array(sample_labels)
    if len(sample_sets.shape) == 2: sample_sets = sample_sets.reshape(1, *sample_sets.shape)
    if len(sample_labels.shape) == 0: sample_labels = sample_labels.reshape(1)
    if sample_sets.shape[0] != sample_labels.shape[0]:
        raise IndexError(f'Must be one label specified for each set of samples ({sample_sets.shape[0]})')
    if sample_sets.shape[2] != len(prior.dists):
        raise IndexError(f'sample_sets must be of the shape (number of sets, number of samples, number of parameters)')
    # Change sample_sets to shape (num_parameters, num_sets, num_samples)
    sample_sets = sample_sets.swapaxes(0, 1)
    sample_sets = sample_sets.swapaxes(0, 2)
    
    fig, axes = create_params_plot(prior, axsize, num_cols)
    twin_axes = []
 
    for ax, sample_set, param, dist, prior_limits in zip(axes, sample_sets, prior.params, prior.dists, prior.limits):
        # Plot samples
        sample_limits = (sample_set.min(), sample_set.max())
#         limits = (min(prior_limits[0], sample_limits[0]), max(prior_limits[1], sample_limits[1]))
        limits = prior_limits
        bins = np.arange(*limits, (limits[1] - limits[0]) / 300)
        for samples, sample_label in zip(sample_set, sample_labels):
            ax.hist(samples, histtype='step', bins=bins, label=sample_label)
            ax.set_ylabel("# Samples")
            ax.legend(loc='upper left')
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
        
        # Plot prior
        x = torch.arange(*limits, (limits[1] - limits[0]) / 300)
        ax_prior = ax.twinx()
        twin_axes.append(ax_prior)
        ax_prior.plot(x, torch.exp(dist.log_prob(x)), ls='--', color='red', label="prior")
        ax_prior.set_ylim(auto=True, bottom=0)
        ax_prior.legend()
        ax_prior.set_yticks([])
        ax_prior.set_ylim(top=ax_prior.get_ylim()[1] * 1.2)
      
    return fig, twin_axes

def plot_means_against_theta(prior, posterior, theta, observations, samples_per_point=1000, confidence_interval=(0.025, 0.975)):
    sample_size = len(posterior.sample(show_progress_bars=False))
    theta_sample_means = np.array([]).reshape((sample_size, 0))
    theta_sample_errors = np.array([]).reshape((sample_size, 2, 0))
    num_sample_sets = observations.shape[0]
    
    for i in range(num_sample_sets):
        print(f'Sampling posterior [{i}/{num_sample_sets}]', end='', flush=True)
        samples = np.array(posterior.sample((samples_per_point,), x=observations[i], show_progress_bars=False)) # (sample_size, samples_per_point)

        means = np.mean(samples, axis=0) # (sample_size,)
        quantiles = scipy.stats.mstats.mquantiles(samples, prob=confidence_interval, axis=0).T # (sample_size, 2)
        errors = np.array([np.abs(q - m) for q, m in zip(quantiles, means)]) # (sample_size, 2)

        theta_sample_means = np.concatenate((theta_sample_means, means.reshape(sample_size, 1)), axis=1)
        theta_sample_errors = np.concatenate((theta_sample_errors, errors.reshape(sample_size, 2, 1)), axis=2)
        print("\r                                  \r", end='')

    print(f'Sampled posterior {num_sample_sets} times.')
    # theta_sample_means is of shape (# of params, # of means calculated)
    # theta_sample_errors is of shape (# of params, 2, # of means calculated)
    
    fig, axes = create_params_plot(prior, axsize=4, num_cols=4)
    for ax, actual, mean, err, param in zip(axes, np.array(theta[:num_sample_sets]).T, theta_sample_means, theta_sample_errors, prior.params):
        ax.set_xlabel("Actual value")
        ax.set_ylabel("Mean of posterior samples")
        ax.errorbar(actual, mean, yerr=err, marker='.', color=(0.3, 0.3, 1), linestyle='', elinewidth=1, ecolor=(1, 0.3, 0.3, 0.2))
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax.plot(lim, lim, ls="--", c=(0, 0, 0, 0.3))
        
    return fig, axes