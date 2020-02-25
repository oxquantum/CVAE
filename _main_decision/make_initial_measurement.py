import numpy as np


def make_inital_measurement(con, options):
    y0, obs_mask, data_index = con.do_experiment_on_grid_and_wait(
        stride=options.DOE_options.stride)  # initial measurement of the grid

    image_shape = options.model_options.image_shape
    obs_data = np.zeros(image_shape)  # full resolution, empty space (pic_shape = (rows, cols))
    obs_data[data_index] = y0  # assign the initial measurement in the full resolution dataspace

    obs_data_mask = np.stack([obs_data, obs_mask], axis=2)  # (rows, cols, 2)
    obs_data_mask = obs_data_mask.astype(np.float32)

    return y0, obs_data, obs_mask, obs_data_mask, data_index
