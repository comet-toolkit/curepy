"""Run MCMC for retrieval"""

"""___Built-In Modules___"""

"""___Third-Party Modules___"""
import os
from multiprocessing import Pool

import emcee
import numpy as np

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
class MCMCRetrieval:
    def __init__(
        self,
        measurement_function,
        observed,
        rand_uncertainty=None,
        cov=None,
        parallel_cores=1,
        uplims=+np.inf,
        downlims=-np.inf,
        b=None,
        u_b=None,
        Sb=None,
        b_iter=0,
        initial_guess=None
    ):
        self.measurement_function = measurement_function
        self.b=None
        self.Sb=None
        self.u_b=None
        if b:
            self.b = np.array(b)
        if Sb:
            self.Sb = np.array(Sb)
        if u_b:
            self.u_b = np.array(u_b)
        self.b_iter = b_iter
        self.observed = observed
        self.rand_uncertainty = np.array(rand_uncertainty)
        if cov is None:
            self.invcov = None
        else:
            self.invcov = np.linalg.inv(np.ascontiguousarray(cov))
        self.uplims = np.array(uplims)
        self.downlims = np.array(downlims)
        self.parallel_cores = parallel_cores
        self.initial_guess=initial_guess

    def measurement_function_x(self,theta):
        x=self.make_x_tuple(theta)
        if self.b is not None:
            xb=x+tuple(self.b)
        else:
            xb=x
        return self.measurement_function(*xb)

    def make_x_tuple(self,theta):
        x=self.initial_guess[:]
        j=0
        for i in range(len(x)):
            if not hasattr(x[i],'__len__'):
                x[i][ii] = theta[j]
                j += 1
            else:
                for ii in range(len(x[i])):
                    if not hasattr(x[i][ii],'__len__'):
                        x[i][ii] = theta[j]
                        j += 1
                    else:
                        for iii in range(len(x[i][ii])):
                            if not hasattr(x[i][ii][iii],'__len__'):
                                x[i][ii][iii] = theta[j]
                                j += 1
                            else:
                                raise ValueError("The initial guess has too high dimensionality.")
        return tuple(x)

    def run_retrieval(self, x_0, nwalkers, steps, burn_in, return_samples=True, return_corr=False):
        self.initial_guess=x_0
        theta_0=np.concatenate(x_0).flatten()
        samples=self.run_MCMC(theta_0,nwalkers,steps,burn_in)
        if self.b is not None:
            b=self.b[:]
            for i in range(self.b_iter):
                for ii in range(len(self.b)):
                    self.b[ii] = np.random.normal() * self.u_b[ii] + b[ii]
                samples = np.vstack(self.run_MCMC(theta_0,nwalkers,steps,burn_in))
                print(i,len(samples))
            self.b = b[:]

        return self.analyse_samples(samples,return_samples,return_corr)

    def run_MCMC(
        self, theta_0, nwalkers, steps, burn_in):
        ndimw = len(theta_0)
        pos = [
            theta_0 * np.random.normal(1.0, 0.1, theta_0.shape)
            + np.random.normal(0.0, 0.001, theta_0.shape)
            for i in range(nwalkers)
        ]

        self.measurement_function_x(theta_0)

        if self.parallel_cores > 1:
            p = Pool(self.parallel_cores)
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob,pool=p)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob)
        sampler.run_mcmc(pos, steps, progress=False)

        samples = sampler.chain[:, :, :].reshape((-1, ndimw))[burn_in::]
        return samples

    def analyse_samples(self,samples,return_samples,return_corr):
        medians = np.median(samples, axis=0)
        unc_up = np.percentile(samples, 84, axis=0) - medians
        unc_down = -(np.percentile(samples, 16, axis=0) - medians)
        unc_avg = (unc_up + unc_down) / 2.0
        corr = np.corrcoef(samples.T)
        print(medians)
        medians = self.make_x_tuple(medians)
        unc_avg = self.make_x_tuple(unc_avg)

        if return_samples:
            if return_corr:
                return medians, unc_avg, corr, samples
            else:
                return medians, unc_avg, samples

        else:
            if return_corr:
                return medians, unc_avg, corr
            else:
                return medians, unc_avg

    def find_chisum(self, theta):
        model = self.measurement_function_x(theta)
        diff = model - self.observed
        if np.isfinite(np.sum(diff)):
            if self.invcov is None:
                return np.sum((diff) ** 2 / self.rand_uncertainty ** 2)
            else:
                # print(diff,np.linalg.inv(self.cov),np.dot(np.dot(diff.T,self.invcov),diff))
                return np.dot(np.dot(diff.T, self.invcov), diff)
        else:
            return np.inf

    def lnlike(self, theta):
        #print(theta,self.find_chisum(theta))
        return -0.5 * (self.find_chisum(theta))

    def lnprior(self, theta):
        if all(self.downlims < theta) and all(self.uplims > theta):
            # if self.syst_uncertainty[0] is None:
            return 0
        # else:
        #     return -0.5*(theta[0]**2/self.syst_uncertainty**2)
        else:
            return -np.inf

    def lnprob(self, theta):
        lp_prior = self.lnprior(theta)
        if not np.isfinite(lp_prior):
            return -np.inf
        lp = self.lnlike(theta)
        return lp_prior + lp
