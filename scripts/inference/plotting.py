import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
import logging
from sbi import analysis

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

def plot_samples_vs_prior(prior, sample_sets, sample_labels, axsize=4, num_cols=4):
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
