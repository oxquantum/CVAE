import time

import numpy as np

from ._helper_functions import estimate_error, get_observation_mask, update_log_prob_mask


def iteration(con, obs_mask, obs_data_mask, obs_data, likelihood, resol_ctrl, acq,
              log_posterior, recon, log_weight_samples, num_obs, options):
    prob = np.exp(log_posterior)

    max_idx = np.argmax(prob)
    x_best_guess = recon[max_idx]

    if num_obs >= options.DOE_options.max_obs:
        print('break')
        # break

    # calculate an acquisition map
    t0 = time.time()
    score = acq.get_score(obs_data_mask, recon, log_weight_samples, 0.0)  # basic

    # code for decision resolution ###
    # measure the error on the current decision grid
    est_err = estimate_error(recon, obs_mask, mask_valid=resol_ctrl.mask_valid, log_prob=log_weight_samples)
    if est_err < 0.05:  # error thresold for increasing resolution
        resol_ctrl.increase()
    # count the number of unobserved locations on the current decision resolution
    num_unobs_decision = resol_ctrl.get_num_unobs(obs_mask)

    # choose next measurement
    if options.DOE_options.batch_mode is False:
        # pointwise selection
        num_obs_next = 1
    else:
        # batch selection
        batch_decision_idxs = options.DOE_options.mcmc_idxs
        next_point = batch_decision_idxs[batch_decision_idxs > num_obs].min()
        next_idx = np.where(batch_decision_idxs == next_point)[0]
        # print('Next point: {}'.format(next_point))
        num_obs_next = next_point - num_obs
    if num_unobs_decision < num_obs_next:
        resol_ctrl.increase()

    next_mask = acq.choose_next_batch(score, num_obs_next, mask_valid=resol_ctrl.mask_valid)

    elapsed_time = time.time() - t0
    print('Time for decision: ', elapsed_time)

    # get the next measurement
    obs = get_observation_mask(next_mask, con)  # mask-based implementaion

    log_weight_samples = update_log_prob_mask(obs, next_mask, recon, log_weight_samples, likelihood)

    # add new observations
    obs_mask = obs_mask + next_mask
    obs_data[next_mask != 0.0] = obs[next_mask != 0.0]
    obs_data_mask = np.stack((obs_data, obs_mask), axis=2)

    saving_data = (prob, x_best_guess, score, next_mask)

    return obs_mask, obs_data, obs_data_mask, log_weight_samples, saving_data
