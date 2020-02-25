import numpy as np

from _main_decision._helper_functions.log_sum_exp import log_sum_exp


class Posterior():
    def __init__(self, net, sess, y0, obs_data_mask, likelihood):
        self.net = net
        self.sess = sess
        self.y0 = y0
        self.observed = obs_data_mask[np.newaxis, ..., :1]
        self.mask = obs_data_mask[np.newaxis, ..., 1:]
        self.likelihood = likelihood

    def reconstruct(self, z, normalize=True, log_mode=False):

        recon, distance = self.net.test_z_with_observed(self.sess, z, self.y0, self.observed, self.mask)
        distance = distance.astype(np.float32)

        log_L = self.likelihood.log_L(distance)
        z_log_prior, _ = self.net.get_prior_density(self.sess, z)  # it returns [log_density, density]
        log_posterior = log_L + z_log_prior
        logZ = log_sum_exp(log_posterior)

        if normalize:
            log_posterior = log_posterior - logZ  # Normalize in log-space

        if log_mode:
            prob = log_posterior
            Z = logZ
        else:
            prob = np.exp(logZ)
            Z = np.exp(logZ)

        return prob, Z, recon

    def __call__(self, z, log_mode=False, normalize=True):
        prob, Z, recon = self.reconstruct(z, log_mode=log_mode, normalize=normalize)
        return prob


