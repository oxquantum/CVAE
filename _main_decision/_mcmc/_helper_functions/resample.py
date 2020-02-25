import time

import numpy as np

from _main_decision._mcmc._helper_functions._helper_functions.sampling import Gaussian_proposal_move, MH_MCMC


def resample(z, likelihood_calc, num_step):
    # input: z, likelihood_calc
    # output: z, log_posterior, recon, log_weight_samples

    move_big = Gaussian_proposal_move(np.square(0.5))
    mcmc_big = MH_MCMC(move_big, likelihood_calc, log_mode=True)
    # move_small = sampling.Gaussian_proposal_move(np.square(0.1))
    # mcmc_small = sampling.MH_MCMC(move_small, likelihood_calc, log_mode=True)

    start_time = time.time()
    z_after_mcmc = mcmc_big(z, num_step=num_step)
    # z_after_mcmc = mcmc_small(z, num_step=100)
    elapsed_time = time.time() - start_time
    print('Time for sampling: ', elapsed_time)
    z = z_after_mcmc

    log_posterior, logZ_posterior, recon = likelihood_calc.reconstruct(z, log_mode=True)  # generate predictions

    # Weights are equal right after resampling
    # (Approximate the posterior distribution of z with equally weighted samples, see the paper)
    batch_size = z.shape[0]
    weight_samples = np.ones(batch_size)
    weight_samples = weight_samples / np.sum(weight_samples)
    log_weight_samples = np.log(weight_samples)
    return z, log_posterior, recon, log_weight_samples
