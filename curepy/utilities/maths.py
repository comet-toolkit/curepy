"""Mathematical methods for retrievals"""

from typing import Union
import numpy as np


def lnlike(cost_function: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert a chi-squared cost to a log likelihood.

    :param cost_function: Chi-squared cost value(s).
    :returns: Log likelihood, equal to ``-0.5 * cost_function``.
    """
    return -0.5 * cost_function
