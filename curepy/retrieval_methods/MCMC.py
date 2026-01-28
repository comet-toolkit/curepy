"""Markov Chain Monte Carlo (MCMC) retrieval class"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.container.measurement_function import MeasurementFunction
from curepy.container.ancillary_parameter import AncillaryParameter
from curepy.container.prior import Prior
from curepy.container.measurement import Measurement

from multiprocessing import Pool
import emcee
import numpy as np

class MCMC(BaseRetrieval):
    """MCMC retrieval object"""
    
    def __init__(
        self,
        measurement_function_obj: MeasurementFunction,
        measurement_obj: Measurement,
        ancillary_obj: AncillaryParameter = None,
        prior_obj: Prior = None,
        progress: bool = True,
    ):
        self.measurement_function_obj = measurement_function_obj
        self.measurement_obj = measurement_obj
        self.ancilary_obj = ancillary_obj
        self.prior_obj = prior_obj
    
    def run_retrieval(self, 
                      nwalkers, 
                      steps, 
                      burn_in,
                      ):
        
        #define theta_0
        theta_0 = self.generate_theta_0(self)
        
        #generate b samples if ancillary data exists
        self.ancilary_obj.generate_b_samples()
        b_samples = self.ancilary_obj.b_samples
        
        #generate samples with MCMC
        if b_samples is None or self.ancillary_obj.b_iter == 1:
            samples = self.run_MCMC(theta_0, nwalkers, steps, burn_in)
        else:
            samples = np.zeros(
                    ((nwalkers * steps - burn_in) * self.b_iter, len(theta_0)),
                    dtype=np.float32,
                )
            b = self.b[:]

            for i in range(len(b_samples[0])):
                for ii in range(len(b_samples)):
                    if b_samples[ii].ndim == 1:
                        self.b[ii] = b_samples[ii][i]
                    elif b_samples[ii].ndim == 2:
                        self.b[ii] = np.array(
                            [b_samples[ii][j][i] for j in range(len(b_samples[ii]))]
                        )
                    else:
                        raise ValueError(
                            "MCMC_retrieval: the dimensionality of one of the parameters in b is not supported (currently the ancillary parameters in b can only be floats or 1d arrays)."
                        )

                samples[
                    i
                    * (nwalkers * steps - burn_in) : (i + 1)
                    * (nwalkers * steps - burn_in),
                    :,
                ] = self.run_MCMC(theta_0, nwalkers, steps, burn_in)

            self.b = b[:]
    
    def run_MCMC(self, theta_0, nwalkers, steps, burn_in):
        ndimw = len(theta_0)
        pos = [self.generate_theta_i(theta_0) for i in range(nwalkers)]
        self.measurement_function_x(theta_0)

        if self.parallel_cores > 1:
            p = Pool(self.parallel_cores)
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob, pool=p)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob)
        sampler.run_mcmc(pos, steps, progress=self.progress)

        samples = sampler.chain[:, :, :].reshape((-1, ndimw))[burn_in::]
        return samples
    
    def generate_theta_0(self):
        
        ig = self.measurement_function_obj.initial_guess
        
        if hasattr(ig, "__len__"):
            if hasattr(ig[0], "__len__"):
                theta_0 = np.concatenate(ig).flatten()
            else:
                theta_0 = np.array(ig).flatten()
        else:
            theta_0 = np.array([ig])
            
        return theta_0
    
    def generate_theta_i(self):
        raise NotImplementedError
    
    def analyse_samples(self, 
                        return_samples=True, 
                        return_corr=False, 
                        include_b_results=False,
                        ):
        raise NotImplementedError
    
    