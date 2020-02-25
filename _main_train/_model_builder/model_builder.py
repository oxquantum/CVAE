from _CVAE import CVAE_type1, CVAE_type2

from ._helper_functions import encoder_pic, encoder_FC, decoder_pic

def build_model(partial_data_shape, model_options):
    # unpacking the flag arguments
    model_name = model_options.model
    latent_size = model_options.latent_size
    batch_size = model_options.batch_size
    do_batch_norm = model_options.do_batch_norm
    activ = model_options.activ
    last_activ = model_options.last_activation
    data_shape = model_options.data_shape

    # Setting VAE type
    if model_name == 'type2' or model_name == 'contextloss' or model_name == 'WGANGP':  # hps of 'contextloss' is from hps of 'type2'
        input_shape_enc = data_shape  # input of the encdoer = full data
        latent_size_dec = latent_size + partial_data_shape[
            0]  # input of the decoder = output of the encoder + initial observations
    else:
        input_shape_enc = partial_data_shape  # input of the encoder = initial observations
        latent_size_dec = latent_size  # latent vector is same for encoder and decoder

    #encoder
    if len(input_shape_enc) == 3: # CNN
        hps_encoder = encoder_pic.HParams(input_shape = input_shape_enc, latent_size = latent_size, 
                                      activ_unit=activ, do_batch_norm=do_batch_norm)
        encoder = encoder_pic(hps_encoder)
    elif len(input_shape_enc) == 1: # Fully-connected
        hps_encoder = encoder_FC.HParams(input_shape=input_shape_enc, layers_units=[128, 256, 256], 
                                         latent_size=latent_size, activ_unit=activ)
        encoder = encoder_FC(hps_encoder)
    #decoder
    hps_decoder = decoder_pic.HParams(latent_size=latent_size_dec, output_shape=data_shape,
                                      activ_unit=activ, last_activ_unit=last_activ, do_batch_norm=do_batch_norm)
    decoder = decoder_pic(hps_decoder)

    #VAE
    if model_name == 'type1':
        hps_vae = CVAE_type1.HParams(batch_size = batch_size, loss_func = 'L1', weight_decay_rate=0.00001)
    elif model_name == 'type2' or model_name == 'contextloss' or model_name == 'WGANGP':  # hps of 'contextloss' is from hps of 'type2'
        hps_vae = CVAE_type2.HParams(batch_size=batch_size, loss_func='L1', weight_decay_rate=0.00001,
                                     zadd_len=partial_data_shape[0])  # Assume that zadd_len is 1-Dimensional
    else:
        raise ValueError('Undefined model')

    return hps_vae, encoder, decoder
