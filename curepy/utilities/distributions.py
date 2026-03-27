"""Distribution functions"""

import numpy as np
from typing import Union


def ln_uniform(
    theta: Union[float, np.ndarray],
    minimum: Union[float, np.ndarray],
    maximum: Union[float, np.ndarray],
) -> float:
    """
    Evaluate the log of a uniform prior distribution.

    Returns ``0`` when all elements of ``theta`` are strictly within
    ``[minimum, maximum]``, and ``-inf`` otherwise.

    :param theta: Current parameter value(s) to evaluate.
    :param minimum: Lower bound(s) of the uniform distribution.
    :param maximum: Upper bound(s) of the uniform distribution.
    :returns: Log probability: ``0`` if in-bounds, ``-numpy.inf`` otherwise.
    """
    if np.all(minimum < theta) and np.all(maximum > theta):
        return 0
    else:
        return -np.inf


def ln_normal(
    theta: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Evaluate the log of an unnormalised normal (Gaussian) prior distribution.

    :param theta: Current parameter value(s) to evaluate.
    :param mu: Mean of the Gaussian distribution.
    :param sigma: Standard deviation of the Gaussian distribution.
    :returns: Log probability proportional to the Gaussian log-density.
    """
    return -0.5 * ((theta - mu) ** 2) / (2 * sigma**2)


def ln_multi_normal(
    theta: np.ndarray,
    mu: np.ndarray,
    Sa_inv: np.ndarray,
) -> float:
    """
    Evaluate the log of an unnormalised multivariate normal prior.

    :param theta: Current parameter vector to evaluate.
    :param mu: Mean vector of the multivariate Gaussian.
    :param Sa_inv: Inverse of the prior covariance matrix.
    :returns: Log probability proportional to the multivariate Gaussian
        log-density.
    """
    diff = theta - mu
    return -0.5 * diff.T @ Sa_inv @ diff
