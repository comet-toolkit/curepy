"""Utilities functions"""

import numpy as np


def flatten_array(A: np.array):
    """
    Flatten array and return flattened array and orginal shape

    :param A: Array to be flattened
    """
    return A.flatten(), A.shape


def reshape_array(A_flat, shape):
    """Reshape flattened array to a given shape

    :param A_flat: Flattened array
    :param shape: Shape for array to be reshaped to
    """
    return A_flat.reshape(shape)


def format_correlation(y, corr):
    """
    Format correlation matrix from class inputs

    :param corr: Correlation matrix input
    :return: Formatted correlation matrix
    """

    if corr is None or not hasattr(y, "__len__"):
        return None
    elif isinstance(corr, str):
        if corr == "rand":
            return np.eye(len(y))
        elif corr == "syst":
            return np.ones((len(y), len(y)))
        else:
            raise ValueError(
                'Error correlation matrix must be defined as None, "rand", "syst", or a custom matrix'
            )
    else:
        return corr


def to_ragged_array(list_of_lists):
    return np.array([np.array(sub) for sub in list_of_lists], dtype=object)
