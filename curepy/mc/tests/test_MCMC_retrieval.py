"""
Tests for mc propagation class
"""

import unittest

import numpy as np
import numpy.testing as npt

import curepy.utilities.utilities as util
from curepy import MCMCRetrieval, plot_corner
import comet_maths as cm

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

class SSG():
    def __init__(self,wavelengths):
        self.wavelengths=wavelengths

    def function(self,central_wav, width, s, amplitude):
        if (central_wav-width>np.min(self.wavelengths)) and (central_wav+width<np.max(self.wavelengths)):
            DN = amplitude*np.exp(-np.abs(self.wavelengths-central_wav)**s/(2*width))
            return DN
        else:
            return 10**6*np.ones_like(self.wavelengths)



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
    def test_class_function(self):
        wavs=np.arange(700,750,0.1)

        ssg=SSG(wavelengths=wavs) #set your wavelengths here

        DN = ssg.function(725,3,2,35000) #set your DN here
        DN_unc = 0.05*np.max(DN)*np.ones_like(DN) #set your DN uncertainties here
        DN = cm.generate_sample(1,DN,DN_unc,"rand")  # noise is added here, your data will alreayd be noisy, so this step not necessary

        retr = MCMCRetrieval(ssg.function,DN,rand_uncertainty=DN_unc,parallel_cores=3,
                             initial_guess=[725,3,2,35000],downlims=[700.,2.,1.5,np.max(DN)*0.9],uplims=[750,10,4,np.max(DN)*1.1],n_input=4)

        medians,unc,samples = retr.run_retrieval(200,200,10*200,return_samples=True,
                                                 return_corr=False)#,x_0=x[0])

        npt.assert_allclose(medians,(725,3,2,35000),rtol=0.01)
        # print(medians,unc)
        # print(samples.shape)
        # plot_corner(samples,"test_corner.png",labels=[r"$\lambda_0$",r"width",r"s",r"A"])


if __name__ == "__main__":
    unittest.main()
