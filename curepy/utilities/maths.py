"""Mathematical methods for retrievals"""

import numpy as np

def find_chisum(self,
                modelled_data,
                observed_data,
                u_observed_data,
                invcov,
                repeat_dims: int = 0):
    diff = modelled_data - observed_data
    if np.isfinite(np.sum(diff)):
        if invcov is None:
            return np.sum((diff) ** 2 / u_observed_data**2)
        else:
            if len(repeat_dims) == 0:
                return np.dot(np.dot(diff.T, invcov), diff)
            elif len(repeat_dims) == 1:
                sum = 0
                for i in range(diff.shape[repeat_dims[0]]):
                    diffi = np.take(diff, i, repeat_dims[0])
                    sum += np.dot(np.dot(diffi.T, invcov), diffi)
                return sum
            else:
                raise ValueError(
                    "Methods for multiple repeat dimensions are not yet implemented,"
                )
    else:
        print(
            "curepy.MCMC_retrieval: the difference between model and observations is infinite"
        )
    return np.inf