import numpy as np

from _main_decision._helper_functions.log_sum_exp import log_sum_exp


def update_log_prob_mask(obs, newobs_mask, recon, log_prob, distance_to_likelihood):
    dist_func = lambda x: np.sum(np.fabs(x))  # L1 distance function
    num_recon = recon.shape[0]
    assert num_recon == log_prob.size
    new_dist = np.zeros(num_recon)

    bool_mask = newobs_mask != 0.0

    for i in range(num_recon):
        recon_i = recon[i, ..., 0]
        diff = recon_i[bool_mask] - obs[bool_mask]
        new_dist[i] = dist_func(diff)
    log_prob = log_prob + distance_to_likelihood(new_dist)  # the function should return log-likelihood
    # normalize
    log_prob = log_prob - log_sum_exp(log_prob)
    return log_prob
