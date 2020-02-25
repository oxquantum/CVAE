def get_observation_mask(mask, con, settletime=None):  # None -> default
    obs = con.do_experiment_mask_pointwise(mask, settletime)
    return obs
