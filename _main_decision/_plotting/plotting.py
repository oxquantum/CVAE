import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np


def do_draw_and_save(obs_data, num_obs, obs_mask, prob, x_best_guess, score, next_mask,
                     options):
    obs_data.dump(str(options.file_options.save_path / 'img_{}.dat'.format(num_obs)))
    obs_mask.dump(str(options.file_options.save_path / 'mask_{}.dat'.format(num_obs)))

    plot_every_step(num_obs, obs_data, obs_mask, options.model_options.image_shape, x_best_guess, prob,
                    options.model_options.batch_size, *options.DOE_options.plot_min_max,
                    options.file_options.save_path)

    score.dump(str(options.file_options.save_path / 'score_{}.dat'.format(num_obs)))
    next_mask.dump(str(options.file_options.save_path / 'mask_next_{}.dat'.format(num_obs)))


def save_and_plot_all(set_data, save_idxs, options):
    # draw and save full data
    print(len(set_data.image_set))
    plt.clf()
    plt.figure(figsize=(25, 25))
    num_save = len(save_idxs)
    num_grid = int(np.ceil(np.sqrt(num_save)))
    for i in range(min(num_save, len(set_data.image_set))):
        plt.subplot(num_grid, num_grid, i + 1)
        # plot with unobserved area black
        plot_with_mask(set_data.image_set[i].reshape(options.model_options.image_shape),
                       set_data.mask_set[i].reshape(options.model_options.image_shape),
                       vmin=options.DOE_options.plot_min_max[0], vmax=options.DOE_options.plot_min_max[1],
                       origin="lower", cmap="seismic")
    plt.savefig(options.file_options.save_path / "decision.png")

    plt.clf()
    plt.figure(figsize=(25, 25))
    plt.imshow(set_data.image_set[-1].reshape(options.model_options.image_shape), vmin=options.DOE_options.plot_min_max[0],
               vmax=options.DOE_options.plot_min_max[1], origin="lower", cmap="seismic")
    plt.savefig(options.file_options.save_path / "final.png")

    plt.clf()
    plt.figure(figsize=(25, 25))
    for i in range(min(num_save, len(set_data.prob_set))):
        plt.subplot(num_grid, num_grid, i + 1)
        plt.plot(np.arange(1, options.model_options.batch_size + 1), set_data.prob_set[i], 'ro')
    plt.savefig(options.file_options.save_path / "prob.png")


def plot_100_recon(predicted, input_shape, plot_min, plot_max, fpath=None):
    assert predicted.shape[0] >= 100

    plt.figure(figsize=(35, 35))
    for i in range(1, 101):
        plt.subplot(10, 10, i)
        plt.imshow(predicted[i - 1].reshape(input_shape), vmin=plot_min, vmax=plot_max, origin="lower", cmap='seismic')
        plt.xticks([])
        plt.yticks([])
    if fpath is None:
        plt.show()
    else:
        plt.savefig(fpath)


def plot_with_mask(data, mask, vmin=-1.0, vmax=1.0, origin="lower", cmap="seismic"):
    cmap = plt.cm.get_cmap(cmap)  # string to object
    colors = Normalize(vmin, vmax, clip=True)(data)
    colors = cmap(colors)
    colors[..., 0:3] = colors[..., 0:3] * mask[..., np.newaxis]
    plt.imshow(colors, origin="lower")
    plt.xticks([])
    plt.yticks([])


def plot_every_step(i, obs_data, obs_mask, pic_shape, x_best_guess, prob, batch_size, plot_min, plot_max, dir_save):
    # plot the data and the best guess
    # plt.clf()
    plt.figure(1)
    plt.clf()
    plt.subplot(1, 2, 1)
    # plt.imshow(obs_data.reshape(pic_shape), vmin=plot_min, vmax=plot_max, origin="lower", cmap="seismic")
    # plot with unobserved area black
    plot_with_mask(obs_data.reshape(pic_shape), obs_mask.reshape(pic_shape),
                   vmin=plot_min, vmax=plot_max, origin="lower", cmap="seismic")

    plt.title("Data")
    # plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(x_best_guess.reshape(pic_shape), vmin=plot_min, vmax=plot_max, origin="lower", cmap="seismic")
    plt.title("Best guess")
    plt.xticks([])
    plt.yticks([])
    # plt.colorbar()
    # plt.show()
    plt.savefig(dir_save / "best_guess_{}.png".format(i))

    ##
    # plt.figure(2)
    # plt.clf()
    # plt.plot(np.arange(1,batch_size+1), prob, 'ro')
    # plt.savefig("results/decision/prob_{}.png".format(i))

    plt.close('all')
