import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

class Subsample(object):
    def __init__(self, stride=1, batch_dim=False, add_noise=None, channel=None):
        self.stride = stride
        self.batch_dim = batch_dim
        self.channel = channel
        self.add_noise = add_noise
    #data: (h, w, c) assumed
    def __call__(self, data):
        if self.batch_dim == True:
            sampled = data[:,::self.stride, ::self.stride]
        else:
            sampled = data[::self.stride, ::self.stride]

        if self.channel is not None:
            sampled = sampled[...,self.channel:self.channel+1]
        if self.add_noise is not None:
            sampled = self.add_noise(sampled)

        # reshape the sampled data
        if self.batch_dim == True:
            batch_size = data.get_shape().as_list()[0]
            sampled = tf.contrib.layers.flatten(sampled)
        else:
            sampled = tf.reshape(sampled, [-1])

        return data, sampled

class Transform_QD_1ch(object):
    def __init__(self, stride=1):
        self.stride = stride
        #self.dim_min = 0.7
        self.dim_min = 0.2
        self.bias_std = 0.1
        self.noise_std = 0.05
    def __call__(self, data):
        #transform data (output of a model)
        # diminishing signal
        dim_factor = tf.clip_by_value(tf.random_uniform([])*1.25,0.0,1.0)#1.0 by 20% chance, 0~1 otherwise
        dim_factor = (1.0-self.dim_min)*dim_factor + self.dim_min #1.0 by 50% chance, dim_min~1 otherwise

        resol = data.get_shape().as_list()[1]
        dim_matrix = tf.linspace(dim_factor,1.0,resol)
        dim_matrix = tf.expand_dims(dim_matrix,0) # height dimension to 1 (multiplication applys to width dim)
        data = data * tf.expand_dims(dim_matrix,2) # expand channel dim

        #random contrast
        #data = data * tf.random_uniform([],0.7,1.3)
        data = data * tf.random_uniform([],0.5,1.5)

        #data = data + tf.random_normal([], stddev=self.bias_std) # add bias
        data = data + tf.random_uniform([],-self.bias_std,self.bias_std)
        data = tf.clip_by_value(data,-1.0,1.0)

        #random flip and multiple -1
        rand_flip = tf.random_uniform([])
        data = tf.cond(rand_flip<0.5, lambda: data, lambda: -1.0*data[::-1])

        #generate input of a model
        sampled = data[::self.stride, ::self.stride]
        sampled = sampled + tf.random_normal(tf.shape(sampled),stddev=self.noise_std)
        sampled = tf.reshape(sampled, [-1])
        return data, sampled


class Prepare_data(object):
    def __init__(self, shape_single, batch_size, shuf_buffer_size=50000, num_threads=1):
        with tf.device("/cpu:0"):
            self.placeholder = tf.placeholder(tf.float32, (None,)+shape_single)
            self.dataset = tf.data.Dataset.from_tensor_slices(self.placeholder)
            self.dataset = self.dataset.shuffle(buffer_size=shuf_buffer_size)

            # two types of transformation for training and testing
            transform_test = Subsample(stride=16, batch_dim=False, channel=0)
            self.dataset_test = self.dataset.map(transform_test, num_parallel_calls=num_threads)
            self.dataset_test = self.dataset_test.batch(batch_size)

            transform_train = Transform_QD_1ch(stride=16)
            # transform_train = Transform_simple(stride=16)
            # transform_train = Identity(stride=16)
            self.dataset_train = self.dataset.map(transform_train, num_parallel_calls=num_threads)
            self.dataset_train = self.dataset_train.batch(batch_size)


            assert self.dataset_train.output_types == self.dataset_test.output_types

            self.iterator = tf.data.Iterator.from_structure(self.dataset_train.output_types,
                                                       self.dataset_train.output_shapes)
            self.shapes = self.dataset_train.output_shapes
            logger.info(self.shapes)

    # expected to be called right before building a model
    def get_next(self):
        return self.iterator.get_next()

    # expected to be called right before training or testing
    def set_data(self, sess, arr, transform_type="train", num_epochs=1, num_consumers=1):
        if transform_type == "train":
            dataset = self.dataset_train
        else:
            dataset = self.dataset_test
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.prefetch(buffer_size=num_consumers)
        init_iter_op = self.iterator.make_initializer(dataset)
        sess.run(init_iter_op, feed_dict={self.placeholder: arr})
