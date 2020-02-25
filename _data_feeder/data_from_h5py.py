from multiprocessing import Pool

import h5py
import numpy as np


class DataShapeError(Exception):
    """Basic exception for if loaded data cannot be reshaped into the required shape"""
    pass


class Data_from_HDF5():

    def __init__(self, training_data_file_list, testing_data_file_list, data_shape):

        self.required_data_shape = data_shape

        self.train = load_data_from_list(training_data_file_list)
        self.test = load_data_from_list(testing_data_file_list)
        self.correct_shape()
        self.any_nan()

    def any_nan(self):
        if self.train is not None:
            assert not np.any(np.isnan(self.train))
        if self.test is not None:
            assert not np.any(np.isnan(self.test))

    def correct_shape(self):

        cases = [('training', self.train), ('testing', self.test)]
        self.train, self.test = tuple(map(lambda case: reshape_data_array(*case, self.required_data_shape), cases))


def reshape_data_array(name, data, required_image_shape):
    if data is not None:
        image_number = data.shape[0]
        try:
            return data.reshape((image_number, *required_image_shape))
        except ValueError:
            image_shape = data.shape[1:]
            message = 'loaded {} data, of shape {}, cannot be reshaped into the required shape {}'.format(name,
                                                                                                          image_shape,
                                                                                                          required_image_shape)
            raise DataShapeError(message)
    else:
        return None


def load_data_from_list(data_file_list):
    if data_file_list:
        with Pool() as p:
            data = list(p.map(load_data_from_file, data_file_list))
        return np.concatenate(data)
    else:
        return None


def load_data_from_file(data_file):
    with h5py.File(data_file, 'r') as f:
        return np.array(f['data']).astype('float32')

