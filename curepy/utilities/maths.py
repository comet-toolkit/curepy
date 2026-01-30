"""Mathematical methods for retrievals"""

import numpy as np
import numdifftools as nd
import warnings

def find_chisum(modelled_data,
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

def lnlike(modelled_data,
            observed_data,
            u_observed_data,
            invcov,
            repeat_dims: int = 0):
    return  -0.5 * (find_chisum(modelled_data, 
                                observed_data, 
                                u_observed_data, 
                                invcov, 
                                repeat_dims))
    
def calculate_Jacobian(fun, x, Jx_diag=False, step=None):
    """
    Calculate the local Jacobian of function y=f(x) for a given value of x

    :param fun: flattened measurement function
    :type fun: function
    :param x: flattened local values of input quantities
    :type x: array
    :param Jx_diag: Bool to indicate whether the Jacobian matrix can be described with semi-diagonal elements. With this we mean that the measurand has the same shape as each of the input quantities and the square jacobain between the measurand and each of the input quantities individually, only has diagonal elements. Defaults to False
    :rtype Jx_diag: bool, optional
    :return: Jacobian
    :rtype: array
    """
    Jfun = nd.Jacobian(fun, step=step)

    if Jx_diag:
        y = fun(x)
        Jfun = nd.Jacobian(fun)
        Jx = np.zeros((len(x), len(y)))
        # print(Jx.shape)
        for j in range(len(y)):
            xj = np.zeros(int(len(x) / len(y)))
            for i in range(len(xj)):
                xj[i] = x[i * len(y) + j]
            # print(xj.shape, xj)
            Jxj = Jfun(xj)
            for i in range(len(xj)):
                Jx[i * len(y) + j, j] = Jxj[0][i]
    else:
        Jx = Jfun(x)

    if len(Jx) != len(fun(x).flatten()):
        warnings.warn(
            "Dimensions of the Jacobian were flipped because its shape "
            "didn't match the shape of the output of the function "
            "(probably because there was only 1 input qty)."
        )
        Jx = Jx.T

    return Jx