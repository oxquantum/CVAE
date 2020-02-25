# the dictionary .get method returns None if the key is not present

class Model_options:

    def __init__(self, model_options_dict):
        if model_options_dict:
            self.model_name = model_options_dict.get('model_name')
            self.latent_size = model_options_dict.get('latent_size')
            self.batch_size = model_options_dict.get('batch_size')
            self.do_batch_norm = model_options_dict.get('do_batch_norm')
            self.activ = model_options_dict.get('activ')
            self.last_activation = model_options_dict.get('last_activation')
            self.data_shape = model_options_dict.get('data_shape')
            self.image_shape = self.data_shape[0:2]

            self.number_of_pixels = self.image_shape[0] * self.image_shape[1]

class Training_options:

    def __init__(self, training_options_dict):
        if training_options_dict:
            self.epochs = training_options_dict.get('epochs')
            self.weight_adv = training_options_dict.get('weight_adv')
            self.display_iter = training_options_dict.get('display_iter')
            self.freeze_on_false = training_options_dict.get('freeze_on_false')
            self.save_iter = training_options_dict.get('save_iter')


class File_options:

    def __init__(self, file_options_dict):
        if file_options_dict:
            self.data_folder = file_options_dict.get('data_folder')
            self.model_folder = file_options_dict.get('model_folder')
            self.name = file_options_dict.get('name')
            self.save_path = file_options_dict.get('save_path')
            self.file_name = file_options_dict.get('file_name')

            training_data_files = file_options_dict.get('training_data_files')

            if training_data_files:
                self.training_data_files = list(map(lambda file: self.data_folder / file, training_data_files))

            self.testing_data_files = file_options_dict.get('testing_data_files')


class Tensorflow_options:

    def __init__(self, tensorflow_options_dict):
        if tensorflow_options_dict:
            self.allow_soft_placement = tensorflow_options_dict.get('allow_soft_placement')
            self.device = tensorflow_options_dict.get('device')
            self.gpu_options = tensorflow_options_dict.get('gpu_options')


class DOE_options:

    def __init__(self, DOE_options_dict):
        if DOE_options_dict:
            self.stride = DOE_options_dict.get('stride')
            self.patch_size = DOE_options_dict.get('patch_size')
            self.distance_scale = DOE_options_dict.get('distance_scale')
            self.mcmc_idxs = DOE_options_dict.get('mcmc_idxs')
            self.num_jumps = DOE_options_dict.get('num_jumps')
            self.batch_mode = DOE_options_dict.get('batch_mode')
            self.max_obs = DOE_options_dict.get('max_obs')
            self.plot_min_max = DOE_options_dict.get('plot_min_max')
            self.num_step = DOE_options_dict.get('num_step')
            self.resolution_control = DOE_options_dict.get('resolution_control')

class Options:

    def __init__(self, options):
        self.model_options = Model_options(options.get('model_options'))
        self.training_options = Training_options(options.get('training_options'))
        self.file_options = File_options(options.get('file_options'))
        self.tensorflow_options = Tensorflow_options(options.get('tensorflow_options'))
        self.DOE_options = DOE_options(options.get('DOE_options'))
