import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def define_flags(flag_options):
    # defining flags
    flags = tf.flags

    # integer flags
    flags.DEFINE_integer('latent_size', flag_options['latent_size'], 'Length of the  latent vector z.')
    flags.DEFINE_integer('epochs', flag_options['epochs'], 'Maximum number of epochs.')
    flags.DEFINE_integer('batch_size', flag_options['batch_size'], 'Mini-batch size for data subsampling.')

    # float flags
    flags.DEFINE_float('weight_adv', flag_options['weight_adv'],
                       'Weight multiplied to the adversarial loss of the generator')

    # string flags
    flags.DEFINE_string('name', flag_options['name'], 'model name suffix')
    flags.DEFINE_string('model', flag_options['model'], 'Model type')
    flags.DEFINE_string('device', flag_options['device'], 'Compute device.')
    flags.DEFINE_string('activ', flag_options['activ'], '?')
    flags.DEFINE_string('last_activation', flag_options['last_activation'], '?')

    # bool flags
    flags.DEFINE_boolean('allow_soft_placement', flag_options['allow_soft_placement'], 'Soft device placement.')
    flags.DEFINE_boolean('do_batch_norm', flag_options['do_batch_norm'], 'do_batch_norm')

    logger.info('Flags created:')
    list(map(lambda item: logger.info('{}:{}'.format(*item)), flag_options.items()))
    logger.info('\n')

    return flags.FLAGS
