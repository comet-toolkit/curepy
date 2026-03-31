"""Utilities functions"""

import numpy as np
from typing import Optional, Union


def flatten_array(A: np.ndarray) -> tuple:
    """
    Flatten array and return the flattened array and the original shape.

    :param A: Array to be flattened.
    :returns: Tuple of ``(A_flat, shape)``.
    """
    return A.flatten(), A.shape


def reshape_array(A_flat: np.ndarray, shape: tuple) -> np.ndarray:
    """Reshape a flattened array to a given shape.

    :param A_flat: Flattened array.
    :param shape: Target shape.
    :returns: Reshaped array.
    """
    return A_flat.reshape(shape)


def format_correlation(
    y: Union[np.ndarray, list],
    corr: Optional[Union[str, np.ndarray]],
) -> Optional[np.ndarray]:
    """
    Format a correlation matrix from user-provided inputs.

    :param y: Reference variable used to determine the length of the
        correlation matrix.
    :param corr: Correlation specification.  Accepted values:

        * ``None`` — no correlation (returns ``None``).
        * ``"rand"`` — random (diagonal identity matrix).
        * ``"syst"`` — fully systematic (all-ones matrix).
        * Custom square ``numpy.ndarray``.
    :returns: Formatted correlation matrix, or ``None``.
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


def to_ragged_array(list_of_lists: list) -> np.ndarray:
    """
    Convert a list of arrays (possibly of different lengths) to a NumPy
    object array.

    :param list_of_lists: List whose elements are sub-arrays or ``None``.
    :returns: 1-D NumPy object array where each element is a
        ``numpy.ndarray`` or ``None``.
    """
    return np.array(
        [np.array(sub) if sub is not None else None for sub in list_of_lists],
        dtype=object,
    )
