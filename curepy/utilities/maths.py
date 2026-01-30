"""Mathematical methods for retrievals"""

import numpy as np
import numdifftools as nd
import warnings

def lnlike(cost_function):
    return  -0.5 * cost_function
    