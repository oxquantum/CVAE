import logging

from ._model_builder import build_model

logger = logging.getLogger(__name__)


def construct_model(preprocessor, options):
    # Model construction
    partial_data_shape = tuple(preprocessor.shapes[1].as_list()[1:])  # shape of partial data except batch_dim
    hps_vae, encoder, decoder = build_model(partial_data_shape=partial_data_shape,
                                            model_options=options.model_options,
                                            )
    logger.info('Model constructed')
    return hps_vae, encoder, decoder
