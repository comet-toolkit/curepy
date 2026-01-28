"""Markov Chain Monte Carlo (MCMC) retrieval class"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.container.retrieval_input import RetrievalInput
from curepy.container.retrieval_result import RetrievalResult

from multiprocessing import Pool
import emcee
import numpy as np

class MCMC(BaseRetrieval):
    """MCMC retrieval object"""
    
    def __init__(
        self,
        nwalkers, 
        steps, 
        burn_in,
        progress: bool = True,
    ):

        self.nwalkers = nwalkers
        self.steps = steps
        self.burn_in = burn_in
        
        self.progress = progress
    
    def run_retrieval(self, 
                      retrieval_input: RetrievalInput,
                      return_samples = False,
                      return_corr = False,
                      return_b_samples = False):
        
        #define theta_0
        theta_0 = self.generate_theta_0(retrieval_input.measurement_function_obj.initial_guess)
        
        #generate b samples if ancillary data exists
        retrieval_input.ancilary_obj.generate_b_samples()
        b_samples = retrieval_input.ancilary_obj.b_samples
        
        #generate samples with MCMC
        if b_samples is None or retrieval_input.ancillary_obj.b_iter == 1:
            samples = self.run_MCMC(theta_0, self.nwalkers, self.steps, self.burn_in)
        else:
            samples = np.zeros(
                    ((self.nwalkers * self.steps - self.burn_in) * retrieval_input.ancillary_obj.b_iter, len(theta_0)),
                    dtype=np.float32,
                )
            
            
            #not refactored yet 
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
                    * (self.nwalkers * self.steps - self.burn_in) : (i + 1)
                    * (self.nwalkers * self.steps - self.burn_in),
                    :,
                ] = self.run_MCMC(theta_0, self.nwalkers, self.steps, self.burn_in)

            self.b = b[:]
            
            #analyse samples, outputs into RetrievalResult
                
    def run_MCMC(self, theta_0, nwalkers, steps, burn_in):
        #not refactored yet
        ndimw = len(theta_0)
        pos = [self.generate_theta_i(theta_0) for i in range(nwalkers)]
        self.measurement_function_x(theta_0)

        if self.parallel_cores > 1:
            p = Pool(self.parallel_cores)
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob, pool=p)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob)
        sampler.run_mcmc(pos, steps, progress=self.progress)

        samples = sampler.get_chain[:, :, :].reshape((-1, ndimw))[burn_in::]
        return samples
    
    @staticmethod
    def generate_theta_0(ig):
        
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
    
    