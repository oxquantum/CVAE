from _CVAE.CVAE_contextloss_model import CVAE_contextloss
from _CVAE.CVAE_type1 import CVAE_type1
from _CVAE.CVAE_type2 import CVAE_type2


def load_CVAE(sess, model_folder, file_name, model_name, batch_size):
    CVAE_dict = {
        'type1': CVAE_type1(),
        'type2': CVAE_type2(),
        'contextloss': CVAE_contextloss()
    }

    net = CVAE_dict.get(model_name)

    if net:
        net.load(sess, model_folder, file_name)
        net.create_testing_nodes(batch_size)
    else:
        raise ValueError('Undefined model')

    return net
