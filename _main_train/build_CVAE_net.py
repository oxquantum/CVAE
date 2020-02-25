import logging

from _CVAE.CVAE_contextloss_model import CVAE_contextloss
from _CVAE.CVAE_type1 import CVAE_type1
from _CVAE.CVAE_type2 import CVAE_type2

logger = logging.getLogger(__name__)


def build_CVAE_net(model_name, hps, encoder, decoder, input_nodes, add_name=""):
    CVAE_dict = {
        'type1': CVAE_type1(),
        'type2': CVAE_type2(),
        'contextloss': CVAE_contextloss()
    }

    net = CVAE_dict.get(model_name)

    if net:
        net.build_net(hps, encoder, decoder, input_nodes, add_name)
    else:
        raise ValueError('Undefined model')

    logger.info('Model created.')

    return net
