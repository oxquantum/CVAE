import logging
import os

import tensorflow as tf

logger = logging.getLogger(__name__)


def train_net(net, sess, model_folder, training_options):
    sess.run(tf.global_variables_initializer())
    net.train(sess,
              model_folder=model_folder,
              display_iter=training_options.display_iter,
              weight_adv=training_options.weight_adv,
              freeze_on=training_options.freeze_on_false,
              save_iter=training_options.save_iter)
    logger.info('Training finished.')

    model_filename = os.path.join(model_folder, net.get_name())
    net.save(sess, model_filename)
    logger.info('Model saved.')
