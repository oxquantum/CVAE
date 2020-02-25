import numpy as np

from _main_decision.Likelihood import Likelihood
from _main_decision._mcmc.do_mcmc import do_mcmc
from _main_decision._measurement_iteration.iteration import iteration
from _main_decision._plotting.plotting import plot_100_recon
from _main_decision._plotting.plotting import save_and_plot_all, do_draw_and_save
from _main_decision.acq_func_gpu import define_acqusition_function
from _main_decision.make_initial_measurement import make_inital_measurement
from _main_decision.resol_control import ResolutionControl_basic, ResolutionControl_double


def choose_resolution_controller(resolution_control):
    # choosing the resolution controller
    cases = {
        'basic': ResolutionControl_basic,  # do nothing with the measurement resolution
        'double': ResolutionControl_double
    }

    ResolutionControl = cases.get(resolution_control)
    if ResolutionControl:
        return ResolutionControl
    else:
        raise ValueError('unacceptable resolution_control option')


class Set_data():

    def __init__(self):
        self.image_set = []
        self.mask_set = []
        self.prob_set = []

    def update(self, obs_data, obs_mask, prob):
        self.image_set.append(np.copy(obs_data))
        self.mask_set.append(np.copy(obs_mask))
        self.prob_set.append(prob)


# Main loop
def DOE(sess, con, net, options):
    # performing the initial measurements
    y0, obs_data, obs_mask, obs_data_mask, data_index = make_inital_measurement(con, options)

    y0 = y0.reshape((-1,))  # to be fed to the reconstruction network
    # sample from the prior distribution of z(z of type2 model does not depend on y0, but type1 does)
    z = net.generate_z(sess, y0)

    # Define likelihood
    likelihood = Likelihood(options.DOE_options.distance_scale)

    # define the acqusition function
    acq = define_acqusition_function(sess, options)

    # Set initial observations to 'visited'
    rows_visited, cols_visited = obs_mask.nonzero()
    rows_visited, cols_visited = rows_visited.tolist(), cols_visited.tolist()
    row_col_list = list(zip(rows_visited, cols_visited))
    acq.check_visited(row_col_list)

    # Define when to save and plot the current state
    jump = options.model_options.number_of_pixels // options.DOE_options.num_jumps

    save_idxs = np.arange(jump, options.model_options.number_of_pixels + 1, jump)
    save_idxs = np.append(save_idxs, [i for i in options.DOE_options.mcmc_idxs if i not in save_idxs])
    save_idxs.sort()

    if options.DOE_options.max_obs is None:
        options.DOE_options.max_obs = options.model_options.number_of_pixels

    # constructing the resolution control
    ResolutionControl = choose_resolution_controller(options.DOE_options.resolution_control)
    resol_ctrl = ResolutionControl(options.model_options.image_shape)

    set_data = Set_data()

    run = 0
    while int(np.sum(obs_mask)) <= options.DOE_options.max_obs:

        run += 1
        num_obs = int(np.sum(obs_mask))
        do_mcmc_bool = num_obs in options.DOE_options.mcmc_idxs

        if do_mcmc_bool:
            z, log_posterior, recon, log_weight_samples = do_mcmc(net, sess, y0, obs_data_mask, likelihood, z, options)

            recon.dump(str(options.file_options.save_path / 'recon_{}.dat'.format(num_obs)))
            plot_100_recon(recon, options.model_options.image_shape, *options.DOE_options.plot_min_max,
                           fpath=options.file_options.save_path / "100_recon_{}".format(num_obs))

        obs_mask, obs_data, obs_data_mask, log_weight_samples, saving_data = iteration(con=con,
                                                                                       obs_mask=obs_mask,
                                                                                       obs_data_mask=obs_data_mask,
                                                                                       obs_data=obs_data,
                                                                                       likelihood=likelihood,
                                                                                       resol_ctrl=resol_ctrl,
                                                                                       acq=acq,
                                                                                       log_posterior=log_posterior,
                                                                                       recon=recon,
                                                                                       log_weight_samples=log_weight_samples,
                                                                                       num_obs=num_obs,
                                                                                       options=options)

        do_draw_and_save_bool = num_obs in save_idxs
        if do_draw_and_save_bool:
            prob, x_best_guess, score, next_mask = saving_data
            do_draw_and_save(obs_data=obs_data,
                             num_obs=num_obs,
                             obs_mask=obs_mask,
                             prob=prob,
                             x_best_guess=x_best_guess,
                             score=score,
                             next_mask=next_mask,
                             options=options)

            set_data.update(obs_data=obs_data, obs_mask=obs_mask, prob=prob)
    save_and_plot_all(set_data, save_idxs, options)
