import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
from torch.distributions.constraints import Constraint
import torch.distributions.constraints as constraints
from torch.distributions import Uniform, Distribution
from .distributions.TruncatedNormal import TruncatedNormal # PyTorch doesn't contain a truncated normal distribution
from .distributions.LogNormalTranslated import LogNormalTranslated
import logging

class JointConstraint(Constraint):
    
    def __init__(self, constraints):
        self._constraints = constraints
        super().__init__()
    
    def check(self, value):
        check = torch.stack([c.check(t) for t, c in zip(value.transpose(0, len(value.shape) - 1), self._constraints)])
        return check.transpose(0, len(check.shape) - 1)

class JointPrior(Distribution):
    
    def __init__(self, dists_dict):
        """
        dists_dict : dict of torch.distributions.Distribution, with the keys as the parameter names
        """
        self.dists = list(dists_dict.values())
        self.params = list(dists_dict.keys())
        self.limits = [dist.icdf(tensor([0.000001, 0.99999])) for dist in self.dists]
        self._batch_shape = torch.Size()
        self._event_shape = torch.Size([len(self.dists)])
        self._mean = tensor([dist.mean for dist in self.dists])
        self._variance = tensor([dist.variance for dist in self.dists])
        self._support = JointConstraint([dist.support for dist in self.dists])
        
    @property
    def mean(self):
        return self._mean
    
    @property
    def variance(self):
        return self._variance
    
    @property
    def support(self):
        return self._support
    
    @property
    def arg_constraints(self):
        return {}
        
    def log_prob(self, x):
        """
        Takes samples of shape (a, b, ..., num_parameters) and returns probabilities in shape (a, b, ...)
        """
        
        if type(x) != torch.Tensor:
            raise ValueError(f'log_prob requires a Tensor, got {type(x)}')
        if x.shape[-1] != len(self.dists):
            raise IndexError(f'log_prob requires size of last dimension of argument to be the same as the number of prior distributions ({len(self.dists)}), got shape {x.shape} instead')

        # Reshape x so we have a simple array of parameter sets, no matter the original dimensionality of x
        new = x.reshape((max(1, np.prod(x.shape[:-1])), len(self.dists)))
        res = torch.zeros(new.shape[0])

        # For each of the parameter sets, calculate the log_prob of each parameter, then sum them to get the joint probability
        for i in range(len(res)):
            theta = new[i]
            logprobs = tensor([dist.log_prob(val) for val, dist in zip(theta, self.dists)])
            res[i] = logprobs.sum()

        # Reshape back to the shape of x (except one scalar value for each parameter set)
        return res.reshape(x.shape[:-1])
    
    def sample(self, shape=torch.Size()):
        return torch.stack([dist.sample(shape) for dist in self.dists], len(shape))

# 'Unnormalised' distributions
distributions = {
    "bottleneck_strength_domestic": TruncatedNormal(a=-0.5, b=np.inf, loc=7500, scale=15000, validate_args=False),
    "bottleneck_strength_wild": TruncatedNormal(a=-0.5, b=np.inf, loc=7500, scale=15000, validate_args=False),
    "bottleneck_time_domestic": TruncatedNormal(a=-6, b=np.inf, loc=3500, scale=500, validate_args=False),
    "bottleneck_time_wild": TruncatedNormal(a=-6, b=np.inf, loc=3500, scale=500, validate_args=False),
    "captive_time": LogNormalTranslated(s=0.4, loc=1, scale=np.exp(2.7), validate_args=False),
    "div_time": TruncatedNormal(a=-7, b=np.inf, loc=40000, scale=4000, validate_args=False),
    "mig_length_post_split": Uniform(low=0, high=10000, validate_args=False),
    "mig_length_wild": LogNormalTranslated(s=0.4, loc=1, scale=np.exp(2.5), validate_args=False),
    "mig_rate_captive": LogNormalTranslated(s=0.5, loc=0, scale=0.08, validate_args=False),
    "mig_rate_post_split": TruncatedNormal(a=0, b=5, loc=0, scale=0.2, validate_args=False),
    "mig_rate_wild": LogNormalTranslated(s=0.5, loc=0, scale=0.08, validate_args=False),
    "pop_size_wild_1": LogNormalTranslated(s=0.2, loc=30, scale=np.exp(8.7), validate_args=False),
    "pop_size_wild_2": LogNormalTranslated(s=0.2, loc=30, scale=np.exp(9), validate_args=False),
    "pop_size_captive": LogNormalTranslated(s=0.5, loc=10, scale=100, validate_args=False),
    "pop_size_domestic_1": LogNormalTranslated(s=0.25, loc=5, scale=np.exp(8.75), validate_args=False),
    "pop_size_domestic_2": LogNormalTranslated(s=0.2, loc=5, scale=np.exp(9.2), validate_args=False)
}

# 'Normalised' distributions (all with loc=0, scale=1)
distributions_normalised = {
    "bottleneck_strength_domestic": TruncatedNormal(a=-0.5, b=np.inf, loc=0, scale=1, validate_args=False),
    "bottleneck_strength_wild": TruncatedNormal(a=-0.5, b=np.inf, loc=0, scale=1, validate_args=False),
    "bottleneck_time_domestic": TruncatedNormal(a=-6, b=np.inf, loc=0, scale=1, validate_args=False),
    "bottleneck_time_wild": TruncatedNormal(a=-6, b=np.inf, loc=0, scale=1, validate_args=False),
    "captive_time": LogNormalTranslated(s=0.4, loc=0, scale=1, validate_args=False),
    "div_time": TruncatedNormal(a=-7, b=np.inf, loc=0, scale=1, validate_args=False),
    "mig_length_post_split": Uniform(low=-0.5, high=0.5, validate_args=False),
    "mig_length_wild": LogNormalTranslated(s=0.4, loc=0, scale=1, validate_args=False),
    "mig_rate_captive": LogNormalTranslated(s=0.5, loc=0, scale=1, validate_args=False),
    "mig_rate_post_split": TruncatedNormal(a=0, b=5, loc=0, scale=1, validate_args=False),
    "mig_rate_wild": LogNormalTranslated(s=0.5, loc=0, scale=1, validate_args=False),
    "pop_size_wild_1": LogNormalTranslated(s=0.2, loc=0, scale=1, validate_args=False),
    "pop_size_wild_2": LogNormalTranslated(s=0.2, loc=0, scale=1, validate_args=False),
    "pop_size_captive": LogNormalTranslated(s=0.5, loc=0, scale=1, validate_args=False),
    "pop_size_domestic_1": LogNormalTranslated(s=0.25, loc=0, scale=1, validate_args=False),
    "pop_size_domestic_2": LogNormalTranslated(s=0.2, loc=0, scale=1, validate_args=False)
}

# The transformations to apply to get from the 'normalised' distributions to the 'unnormalised'
transforms = {
    "bottleneck_strength_domestic": {'loc': 7500, 'scale': 15000},
    "bottleneck_strength_wild": {'loc': 7500, 'scale': 15000},
    "bottleneck_time_domestic": {'loc': 3500, 'scale': 500},
    "bottleneck_time_wild": {'loc': 3500, 'scale': 500},
    "captive_time": {'loc': 1, 'scale': np.exp(2.7)},
    "div_time": {'loc': 40000, 'scale': 4000},
    "mig_length_post_split": {'loc': 5000, 'scale': 10000},
    "mig_length_wild": {'loc': 1, 'scale': np.exp(2.5)},
    "mig_rate_captive": {'loc': 0, 'scale': 0.08},
    "mig_rate_post_split": {'loc': 0, 'scale': 0.2},
    "mig_rate_wild": {'loc': 0, 'scale': 0.08},
    "pop_size_wild_1": {'loc': 30, 'scale': np.exp(8.7)},
    "pop_size_wild_2": {'loc': 30, 'scale': np.exp(9)},
    "pop_size_captive": {'loc': 10, 'scale': 100},
    "pop_size_domestic_1": {'loc': 5, 'scale': np.exp(8.75)},
    "pop_size_domestic_2": {'loc': 5, 'scale': np.exp(9.2)}
}

def join_priors(normalise=False):
    return JointPrior(distributions_normalised if normalise else distributions)

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
 
    for ax, sample_set, param, dist, limits in zip(axes, sample_sets, prior.params, prior.dists, prior.limits):
        # Plot samples
        bins = np.arange(*limits, (limits[1] - limits[0]) / 300)
        for samples, sample_label in zip(sample_set, sample_labels):
            ax.hist(samples, histtype='step', bins=bins, label=sample_label)
            ax.set_ylabel("# Samples")
            ax.legend(loc='upper left')
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
        
        # Plot prior
        x = torch.arange(*limits, (limits[1] - limits[0]) / 300)
        ax_prior = ax.twinx()
        ax_prior.plot(x, torch.exp(dist.log_prob(x)), ls='--', color='red', label="prior")
        ax_prior.set_ylim(auto=True, bottom=0)
        ax_prior.legend()
        ax_prior.set_yticks([])
        ax_prior.set_ylim(top=ax_prior.get_ylim()[1] * 1.2)
      
    return fig, axes
