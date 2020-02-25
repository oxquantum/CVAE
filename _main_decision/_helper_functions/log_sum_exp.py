import numpy as np


def log_sum_exp(logvals):
    # logvals: 1D assumed
    assert logvals.ndim == 1
    maxidx = np.argmax(logvals)
    adjust = logvals[maxidx]
    exp_terms = np.exp(logvals - adjust)
    exp_terms[maxidx] = 0.0  # remove 1.0, and use log1p for better numerical stability
    return adjust + np.log1p(np.sum(exp_terms, axis=None))
