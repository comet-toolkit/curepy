"""Distribution functions"""

import numpy as np


def ln_uniform(theta, minimum, maximum):
    if np.all(minimum < theta) and np.all(maximum > theta):
        return 0

    else:
        return -np.inf


def ln_normal(theta, mu, sigma):
    return -0.5 * ((theta - mu) ** 2) / (2 * sigma**2)


def ln_multi_normal(theta, mu, Sa_inv):
    diff = theta - mu
    return -0.5 * diff.T @ Sa_inv @ diff
