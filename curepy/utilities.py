"""Utilities functions"""

import numpy as np

def flatten_array(
    A:np.array
    ):
    """
    Flatten array and return flattened array and orginal shape

    :param A: Array to be flattened
    """
    return A.flatten(), A.shape

def reshape_array(A_flat,shape):
    """Reshape flattened array to a given shape

    :param A_flat: Flattened array
    :param shape: Shape for array to be reshaped to
    """
    return A_flat.reshape(shape)
    