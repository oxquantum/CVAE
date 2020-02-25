from _main_decision._mcmc._helper_functions.Posterior import Posterior
from _main_decision._mcmc._helper_functions.resample import resample


def do_mcmc(net, sess, y0, obs_data_mask, likelihood, z, options):
    # MCMC move during experiment
    # MCMC move settings
    likelihood_calc = Posterior(net, sess, y0, obs_data_mask,
                                likelihood)  # function that returns likelihood
    z, log_posterior, recon, log_weight_samples = resample(z, likelihood_calc, options.DOE_options.num_step)

    return z, log_posterior, recon, log_weight_samples
