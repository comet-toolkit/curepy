"""Distribution functions"""

import numpy as np

def ln_uniform(theta, minimum, maximum):
    if np.all(
        minimum < theta
        ) and np.all(
        maximum > theta
    ):
            return 0
        
    else:
        return -np.inf