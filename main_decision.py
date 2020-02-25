import datetime
import logging
from pathlib import Path

import tensorflow as tf

from _data_feeder import DataFromNPY
from _helper_functions import create_directory
from _main_decision import DOE, load_CVAE
from _options import Options

# configure the logger
logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler('main_training_logfile.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)


options_dict = {
    'model_options': {
        'model_name': 'contextloss',
        'batch_size': 100,
        'data_shape': (128, 128, 1),
        'shape_data': (-1,),  # make a initial measurement (y0) a vector
    },

    'file_options': {
        'model_folder': Path('./models'),  # path for tensorflow reconstruction model
        'file_name': 'CVAE_closs_128_latent100mixed_epoch_30_',

        'testing_data_files': Path(
            "./test_data/['9.0000000000', '10.0000000000', '0.2608695652', '0.0200000000', -1].npy"),

        'save_path': Path('./results/decision') / '{date:%Y_%m_%d_%H_%M}'.format(date=datetime.datetime.now())
    },

    'tensorflow_options': {
        'device': "/gpu:0",
        "allow_soft_placement": True,
        'gpu_options': tf.GPUOptions(allow_growth=True)
    },

    'DOE_options': {
        'stride': 16,
        'patch_size': 1,
        'distance_scale': 1,
        'mcmc_idxs': [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        'num_jumps': 64,
        'batch_mode': False,
        'plot_min_max': (-1, 1),
        'num_step': 1,
        'resolution_control': 'basic'
    }
}

options = Options(options_dict)

create_directory(options.file_options.save_path)  # creating the save directory if it does not exist

with tf.device(options.tensorflow_options.device):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=options.tensorflow_options.allow_soft_placement,
                                          gpu_options=options.tensorflow_options.gpu_options)) as sess:
        # build a reconstruction model and load variables
        net = load_CVAE(sess=sess,
                        model_folder=options.file_options.model_folder,
                        model_name=options.model_options.model_name,
                        batch_size=options.model_options.batch_size,
                        file_name=options.file_options.file_name)

        con = DataFromNPY(options.file_options.testing_data_files,
                          shape_out=options.model_options.image_shape,
                          upside_down=False,
                          contrast=1.0)

        DOE(sess, con, net, options=options)
