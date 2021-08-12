import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.distributions.constraints as constraints
from torch.distributions import Uniform, Distribution
from snle.distributions.TruncatedNormal import TruncatedNormal # PyTorch doesn't contain a truncated normal distribution
from snle.distributions.LogNormalTranslated import LogNormalTranslated
import logging

class JointPrior(Distribution):
    
    def __init__(self, dists):
        """
        dists : list of torch.distributions.Distribution
        """
        self.dists = dists
        self._batch_shape = torch.Size()
        self._event_shape = torch.Size([len(dists)])
        self._mean = Tensor([dist.mean for dist in dists])
        self._variance = Tensor([dist.variance for dist in dists])
        self._support = constraints.stack([dist.support for dist in dists], dim=0)
        
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
        
        if type(x) != Tensor:
            raise ValueError(f'log_prob requires a Tensor, got {type(x)}')
        if x.shape[-1] != len(self.dists):
            raise IndexError(f'log_prob requires size of last dimension of argument to be the same as the number of prior distributions ({len(self.dists)}), got shape {x.shape} instead')

        # Reshape x so we have a simple array of parameter sets, no matter the original dimensionality of x
        new = x.reshape((max(1, np.prod(x.shape[:-1])), len(self.dists)))
        res = torch.zeros(new.shape[0])

        # For each of the parameter sets, calculate the log_prob of each parameter, then sum them to get the joint probability
        for i in range(len(res)):
            theta = new[i]
            logprobs = Tensor([dist.log_prob(val) for val, dist in zip(theta, self.dists)])
            res[i] = logprobs.sum()

        # Reshape back to the shape of x (except one scalar value for each parameter set)
        return res.reshape(x.shape[:-1])
    
    def sample(self, shape=torch.Size()):
        return torch.stack([dist.sample(shape) for dist in self.dists], len(shape))

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

def join_priors(dists):
    return JointPrior(list(dists.values()))

def get_prior_limits(param):
    return distributions[param].icdf(Tensor([0.000001, 0.99999]))

def plot_all_priors(dists):
    num_plots = len(dists)
    num_cols = 4
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, constrained_layout=True, figsize=(4 * num_cols, 4 * num_rows))
    for ax, dist, param in zip(axes.flat, dists.values(), dists.keys()):
        ax.set_title('{0}\nmean={1:.2G}, var={2:.2G}'.format(param, dist.mean, dist.variance))
#         ax.set_title(param)
        minmax = get_prior_limits(param)
        x = torch.arange(*minmax, (minmax[1] - minmax[0]) / 500)
        ax.plot(x, torch.exp(dist.log_prob(x)), label="pdf")
        ax.axvline(dist.mean, ls='--', lw=1, c=(0, 0, 0, 0.5), label="mean")
        ax.axvspan(dist.mean - np.sqrt(dist.variance), dist.mean + np.sqrt(dist.variance), color=(1, 0, 0, 0.1), label="+/- sigma")
        ax.legend()
    plt.show()

def main():
    plot_all_priors()

if __name__ == "__main__":
    main()
