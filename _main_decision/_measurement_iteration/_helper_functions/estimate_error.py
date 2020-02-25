import numpy as np
from scipy import ndimage


def estimate_error(recon, mask, mask_valid=None, log_prob=None):
    vals = list()
    for i in range(recon.shape[0]):
        var_map = calc_error_map(recon[i, ..., 0])
        unobs_mask = mask == 0.0
        if mask_valid is not None:
            unobs_mask = np.logical_and(unobs_mask, mask_valid)
        val = np.sum(var_map[unobs_mask])  # calculate the remaining variance
        var_tot = np.sum(var_map[mask_valid]) if mask_valid is not None else np.sum(var_map)
        val = val / var_tot
        vals.append(val)
    if log_prob is None:
        est_error = np.mean(vals)
    else:
        prob = np.exp(log_prob)
        prob = prob / np.sum(prob)
        est_error = np.sum(prob * np.array(vals))
    return est_error


def calc_error_map(img):
    # Derivative is not scaled for simplicity of this demo code
    result = euclidean_dist(ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1))
    return result


def euclidean_dist(x, y):
    return np.sqrt(np.square(x) + np.square(y))
