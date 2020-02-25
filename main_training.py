import logging
from pathlib import Path

import tensorflow as tf

from _data_feeder import Data_from_HDF5, Prepare_data
from _helper_functions import create_directory
from _main_train import build_CVAE_net, train_net, construct_model
from _options import Options

# configure the logger
logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('main_training_logfile.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

options_dict = {
    'model_options': {
        'model': 'contextloss',
        'latent_size': 100,
        'batch_size': 100,
        'do_batch_norm': True,
        'activ': 'elu',
        'last_activation': 'tanh',
        'data_shape': (128, 128, 1)  # shape of single data, also output shape of the decoder
    },
    'training_options': {
        'epochs': 30,
        'weight_adv': 0.0,
        'display_iter': 100,
        'freeze_on_false': False,
        'save_iter': 100000
    },

    'file_options': {
        'data_folder': Path('./data'),
        'model_folder': Path('./models'),

        'training_data_files': [
            '20200113-143735.h5py'
        ],

        'name': '',
    },

    'tensorflow_options': {
        'allow_soft_placement': True,
        'device': '/gpu:0',
        'gpu_options': tf.GPUOptions(allow_growth=True)
    }
}

# convert the dict to named tuple
options = Options(options_dict)

# creating the directory to save the models
create_directory(directory_path=options.file_options.model_folder)

# load the whole HDF5 data into memory
data_set = Data_from_HDF5(training_data_file_list=options.file_options.training_data_files,
                          testing_data_file_list=None,
                          data_shape=options.model_options.data_shape
                          )  # load all data into memory

logger.info('Data from HDF5 loaded')

preprocessor = Prepare_data(options.model_options.data_shape, options.model_options.batch_size)
logger.info('Preprocessor constructed')
hps_vae, encoder, decoder = construct_model(preprocessor, options)
logger.info('Model built')

with tf.device(options.tensorflow_options.device):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=options.tensorflow_options.allow_soft_placement,
                                          gpu_options=options.tensorflow_options.gpu_options
                                          )) as sess:
        input_nodes = preprocessor.get_next()  # pair of (full, partial) data expected

        add_model_name = 'mixed_epoch_{}_{}'.format(options.training_options.epochs,
                                                    options.file_options.name)  # will be added to file names

        net = build_CVAE_net(model_name=options.model_options.model,
                             hps=hps_vae,
                             encoder=encoder,
                             decoder=decoder,
                             input_nodes=input_nodes,
                             add_name=add_model_name)
        logger.info('Net built')

        preprocessor.set_data(sess=sess,
                              arr=data_set.train,
                              transform_type='train',
                              num_epochs=options.training_options.epochs)
        logger.info(
            'Data loaded to preprocessor, beginning training - epochs: {}'.format(options.training_options.epochs))
        train_net(net=net,
                  sess=sess,
                  model_folder=options.file_options.model_folder,
                  training_options=options.training_options)
        logger.info('Training complete')
