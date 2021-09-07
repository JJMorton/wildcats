import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
from torch.distributions.constraints import Constraint
import torch.distributions.constraints as constraints
from torch.distributions import Uniform, Distribution
from .truncnorm.TruncatedNormal import TruncatedNormal # PyTorch doesn't contain a truncated normal distribution
from .lognorm.LogNormalTranslated import LogNormalTranslated
import logging

class JointConstraint(Constraint):
    
    def __init__(self, constraints):
        self._constraints = constraints
        super().__init__()
    
    def check(self, value):
        # value of shape (n1, n2, 16)
        # constraints of shape (16,)
        # t of shape (n1, n2, 1), squeeze() gives (n1, n2)
        # each check is of shape (n1, n2)
        # result is of shape (n1, n2)
        
        checks = [c.check(t.squeeze(t.dim() - 1)) for t, c in zip(value.split(1, value.dim() - 1), self._constraints)]
        res = checks[0]
        for check in checks:
            res = res & check
        return check

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

        res = sum([dist.log_prob(val) for dist, val in zip(self.dists, x.split(1, x.dim() - 1))])
        return res.squeeze(res.dim() - 1)
    
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
    # Original:
#     "mig_length_wild": LogNormalTranslated(s=0.4, loc=1, scale=np.exp(2.5), validate_args=False),
    # Widen 1:
#     "mig_length_wild": LogNormalTranslated(s=0.5, loc=0, scale=np.exp(3.2), validate_args=False),
    # Widen 2:
    "mig_length_wild": LogNormalTranslated(s=0.5, loc=0, scale=80, validate_args=False),
    "mig_rate_captive": LogNormalTranslated(s=0.5, loc=0, scale=0.08, validate_args=False),
    "mig_rate_post_split": TruncatedNormal(a=0, b=5, loc=0, scale=0.2, validate_args=False),
    "mig_rate_wild": LogNormalTranslated(s=0.5, loc=0, scale=0.08, validate_args=False),
    "pop_size_wild_1": LogNormalTranslated(s=0.2, loc=30, scale=np.exp(8.7), validate_args=False),
    "pop_size_wild_2": LogNormalTranslated(s=0.2, loc=30, scale=np.exp(9), validate_args=False),
    "pop_size_captive": LogNormalTranslated(s=0.5, loc=10, scale=100, validate_args=False),
    "pop_size_domestic_1": LogNormalTranslated(s=0.25, loc=5, scale=np.exp(8.75), validate_args=False),
    "pop_size_domestic_2": LogNormalTranslated(s=0.2, loc=5, scale=np.exp(9.2), validate_args=False)
}

def join_priors(normalise=False):
    return JointPrior(distributions_normalised if normalise else distributions)

def get_param_names():
    return list(distributions.keys())