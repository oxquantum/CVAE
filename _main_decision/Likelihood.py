import numpy as np


class Likelihood(object):
    def __init__(self, distance_scale):
        self.distance_scale = distance_scale

    def log_L(self, distance):
        return -(distance / self.distance_scale)

    def exp_adjusted(self, log_L):
        log_L_ = log_L - np.amax(log_L)  # for numerical stability (to prevent underflow)
        return np.exp(log_L_)

    def __call__(self, dist):
        log_likelihood = self.log_L(dist)
        return self.exp_adjusted(log_likelihood)
