"""Run MCMC for retrieval"""

"""___Built-In Modules___"""

"""___Third-Party Modules___"""
import os

import numpy as np
from scipy.optimize import minimize

import punpy.utilities.utilities as util

os.environ["OMP_NUM_THREADS"] = "1"

"""___NPL Modules___"""

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/03/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

inds_cache = {}


# lock = threading.Lock()
class LPURetrieval:
    def __init__(
        self,
        measurement_function,
        observed,
        syst_uncertainty=None,
        rand_uncertainty=None,
        cov=None,
        uplims=+np.inf,
        downlims=-np.inf,
        Jx=None,
    ):
        """
        Initialise Law of Propagation of Uncertainty retrieval

        :param measurement_function:
        :type measurement_function:
        :param observed:
        :type observed:
        :param syst_uncertainty:
        :type syst_uncertainty:
        :param rand_uncertainty:
        :type rand_uncertainty:
        :param cov:
        :type cov:
        :param uplims:
        :type uplims:
        :param downlims:
        :type downlims:
        """
        self.measurement_function = measurement_function
        self.observed = observed
        self.rand_uncertainty = np.array([rand_uncertainty])
        self.syst_uncertainty = np.array([syst_uncertainty])
        if cov is None:
            self.invcov = cov
        else:
            self.invcov = np.linalg.inv(np.ascontiguousarray(cov))
            # print(observed,cov,self.invcov)
        self.Jx = Jx
        self.uplims = np.array(uplims)
        self.downlims = np.array(downlims)

    def run_retrieval(self, theta_0, return_corr=True):
        res = minimize(self.find_chisum, theta_0)
        if self.Jx is None:
            Jx = util.calculate_Jacobian(self.measurement_function, res.x)
        else:
            Jx = self.Jx

        return tuple(res.x) + tuple(self.process_inverse_jacobian(Jx, return_corr))

    def process_inverse_jacobian(self, J, return_corr=True):
        covx = self.calculate_measurand_covariance(J, self.invcov)
        u_func = np.diag(covx) ** 0.5
        corr_x = util.convert_cov_to_corr(covx, u_func)
        if return_corr:
            return u_func, corr_x
        else:
            return u_func

    def calculate_measurand_covariance(self, J, Sy_inv, Sa_inv=None, Sb_inv=None):
        Se_inv = Sy_inv + Sb_inv
        if Sa_inv:
            return np.linalg.inv(np.dot(np.dot(J.T, Se_inv), J) + Sa_inv)
        else:
            return np.linalg.inv(np.dot(np.dot(J.T, Se_inv), J))

    def find_chisum(self, theta):
        model = self.measurement_function(theta)
        diff = model - self.observed
        if np.isfinite(np.sum(diff)):
            if self.invcov is None:
                return np.sum((diff) ** 2 / self.rand_uncertainty**2)
            else:
                return np.dot(np.dot(diff.T, self.invcov), diff)
        else:
            return np.inf
