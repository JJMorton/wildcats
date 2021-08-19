import numpy as np
from torch.distributions import TransformedDistribution, LogNormal
from torch.distributions.transforms import AffineTransform
from torch.distributions.constraints import greater_than as GreaterThan

class LogNormalTranslated(TransformedDistribution):
    """
    Translated LogNormal distribution with mean and variance defined
    """
    def __init__(self, s, loc, scale, validate_args=None):
        """
        Parameters `s`, `loc` and `scale` are the same as those given to scipy.stats.lognorm
        """
        base_dist = LogNormal(np.log(scale), s, validate_args=validate_args)
        super(LogNormalTranslated, self).__init__(base_dist, AffineTransform(loc=loc, scale=1), validate_args=validate_args)
        self._loc = base_dist.loc + loc
        self._mean = base_dist.mean + loc
        self._variance = base_dist.variance
        self._support = GreaterThan(lower_bound = loc)
        
    @property
    def loc(self):
        return self._loc
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def variance(self):
        return self._variance
    
    @property
    def support(self):
        return self._support
