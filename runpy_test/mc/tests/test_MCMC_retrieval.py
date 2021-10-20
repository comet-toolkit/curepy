"""
Tests for mc propagation class
"""

import unittest

import numpy as np
import numpy.testing as npt

import runpy_test.utilities.utilities as util
from runpy_test.mc.MCMC_retrieval import MCMCRetrieval

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "14/4/2020"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def function(x1):
    return 3*x1**2

x= np.ones(50)*10
x_err = np.ones(50)
y = function(x)
y_err = np.ones(50)*60

wavs=np.arange(500,1000,50)


def function_b(x1, x2):
    return x1 ** 2 - 10 * x2 + 30.0

x1_b = np.ones(200) * 10
x2_b = np.ones(200) * 30
x1err_b = np.ones(200)
x2err_b = 2 * np.ones(200)

y_b=function_b(x1_b,x2_b)

# below, the higher order Taylor expansion terms have been taken into account, and amount to 2.
yerr_b_uncorr = 802 ** 0.5 * np.ones(200)
yerr_b_corr = 2 ** 0.5 * np.ones(200)


class TestMCMCRetrieval(unittest.TestCase):
    """
    Class for unit tests
    """

    def test_single_inputqty_0D(self):
        retr = MCMCRetrieval(function,y[0],rand_uncertainty=y_err[0],parallel_cores=20,
                             initial_guess=x[0]+1,downlims=0.)
        medians,unc,samples = retr.run_retrieval(150,300,20,return_samples=True,
                                                 return_corr=False)#,x_0=x[0])

        npt.assert_allclose(medians[0],x[0],rtol=0.01)
        npt.assert_allclose(unc[0],x_err[0],rtol=0.05)

    def test_single_inputqty_1D(self):
        retr = MCMCRetrieval(function,y,rand_uncertainty=y_err,parallel_cores=20,
                             initial_guess=x,n_input=1)
        medians,unc,samples = retr.run_retrieval(200,800,200,return_samples=True,
                                                 return_corr=False)#,x_0=x)
        print(medians,unc)

        # npt.assert_allclose(medians[0],x,rtol=0.01)
        # npt.assert_allclose(unc[0],x_err,rtol=0.05)

    def test_multiple_inputqty_1D(self):
        retr = MCMCRetrieval(function_b,y_b[0],rand_uncertainty=yerr_b_uncorr[0],
                             parallel_cores=20,initial_guess=[x1_b[0],x2_b[0]],n_input=2)
        medians,unc,samples = retr.run_retrieval(200,800,200,return_samples=True,
                                                 return_corr=False)#,x_0=[x1_b[0],x2_b[0]])
        print(medians,unc)

        # npt.assert_allclose(medians[0],x1_b[0],rtol=0.01)
        # npt.assert_allclose(medians[1],x2_b[0],rtol=0.01)

     # def test_multiple_inputqty_1D(self):
     #        retr = MCMCRetrieval(function_b,y_b[0],rand_uncertainty=yerr_b_uncorr,parallel_cores=20,initial_guess=[x1_b,x2_b],n_input=2)
     #        medians,unc,samples=retr.run_retrieval(200 , 800, 200, return_samples=True, return_corr=False, x_0=y_1)
     #        print(medians,unc)
     #
     #        npt.assert_allclose(medians[0],x_1,rtol=0.01)
     #        npt.assert_allclose(unc[0],x_1err,rtol=0.05)


if __name__ == "__main__":
    unittest.main()
