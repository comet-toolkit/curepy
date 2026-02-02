"""Markov Chain Monte Carlo (MCMC) retrieval class"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.container.retrieval_input import RetrievalInput
from curepy.container.retrieval_result import RetrievalResult
from curepy.utilities.maths import lnlike

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
        parallel_cores: int = 1,
    ):

        self.nwalkers = nwalkers
        self.steps = steps
        self.burn_in = burn_in
        
        self.progress = progress
        self.parallel_cores = parallel_cores
    
    def run_retrieval(self, 
                      retrieval_input: RetrievalInput,
                      return_samples = False,
                      return_corr = False,
                      return_b_samples = False):
        
        #format and define retrieval input
        if retrieval_input.ancillary_obj is None:
            retrieval_input.build_ancillary()
        if retrieval_input.prior_obj is None:
            retrieval_input.build_prior(prior_shape="uniform",
                                        prior_params={"minimum": -np.inf,
                                                      "maximum": np.inf})    
        self.retrieval_input = retrieval_input
        
        #define theta_0
        theta_0 = self.generate_theta_0(self.retrieval_input.measurement_function_obj.initial_guess)
        
        #generate b samples if ancillary data exists
        self.retrieval_input.ancillary_obj.generate_b_samples()
        b_samples = self.retrieval_input.ancillary_obj.b_samples
        
        #generate samples with MCMC
        if b_samples is None or self.retrieval_input.ancillary_obj.b_iter == 1:
            samples = self.run_MCMC(theta_0, self.nwalkers, self.steps, self.burn_in)
        else:
            samples = np.zeros(
                    ((self.nwalkers * self.steps - self.burn_in) * self.retrieval_input.ancillary_obj.b_iter, len(theta_0)),
                    dtype=np.float32,
                )
            
            
            #todo: not refactored yet 
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
            
        return self.analyse_samples(
        samples, b_samples, return_samples, return_corr, return_b_samples
    )
                
    def run_MCMC(self, theta_0, nwalkers, steps, burn_in):
        #todo: not refactored yet
        ndimw = len(theta_0)
        pos = [self.generate_theta_i(theta_0) for i in range(nwalkers)]
        #self.measurement_function_x(theta_0) ##todo: commented out in refactor, delete if examples pass without using

        if self.parallel_cores > 1:
            p = Pool(self.parallel_cores)
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob, pool=p)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob)
        sampler.run_mcmc(pos, steps, progress=self.progress)

        samples = sampler.get_chain()[:, :, :].reshape((-1, ndimw))[burn_in::]
        return samples
    
    def generate_theta_i(self, theta_0, factor_std=0.1):
        theta_i = theta_0 * np.random.normal(1.0, factor_std, theta_0.shape)
        if np.all(self.retrieval_input.prior_obj.prior_params["minimum"] < theta_i) and np.all(
            self.retrieval_input.prior_obj.prior_params["maximum"] > theta_i#todo: check what to do for non uniform priors
        ):
            return theta_i
        else:
            return self.generate_theta_i(theta_0, factor_std=factor_std * 0.9)
    
    def analyse_samples(
        self, samples, b_samples, return_samples, return_corr, return_b_samples
    ):
        
        medians = np.median(samples, axis=0)
        
        #todo: still need to refactor/finalise calculations
        unc_up = np.percentile(samples, 84, axis=0) - medians
        unc_down = -(np.percentile(samples, 16, axis=0) - medians)
        unc_avg = (unc_up + unc_down) / 2.0

        if return_corr:
            if samples.shape[1] > 1:
                corr = np.corrcoef(samples.T)
            else:
                corr = np.ones((1,))

        outs = RetrievalResult(x = medians,
                               u_x = unc_avg,
                               corr_x = corr if return_corr else None,
                               samples = samples if return_samples else None,
                               b_samples = b_samples if return_b_samples else None,
                               )

        return outs
    
    def lnprob(self, theta):
        lp_prior = self.retrieval_input.prior_obj.lnprior(
            theta,
            **self.retrieval_input.prior_obj.prior_params)
        if not np.isfinite(lp_prior):
            return -np.inf
        
        lp = lnlike(self.find_chisum(theta,
                                     repeat_dims=[]))#todo: placeholder! figure out where to define
        
        return lp_prior + lp
    
    