"""Run MCMC for retrieval"""

"""___Built-In Modules___"""

"""___Third-Party Modules___"""
import os
from multiprocessing import Pool
import copy

import emcee
import numpy as np

np.random.seed(42)

os.environ["OMP_NUM_THREADS"] = "1"

"""___NPL Modules___"""
import punpy
import comet_maths as cm

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
        syst_uncertainty=None,
        cov=None,
        parallel_cores=1,
        uplims=+np.inf,
        downlims=-np.inf,
        b=None,
        u_b=None,
        corr_b=None,
        b_corr_between=None,
        b_iter=0,
        b_samples=None,
        initial_guess=None,
        n_input=None,
        progress=True,
        repeat_dims=[],
        circular=False,
    ):
        self.measurement_function = measurement_function
        self.b = None
        self.u_b = None
        self.corr_b = None
        self.b_corr_between = None
        self.circular = circular
        if b:
            try:
                self.b = np.array(b)
            except:
                self.b = np.array(b, dtype=object)
        if u_b:
            self.u_b = np.array(u_b)
        if corr_b:
            self.corr_b = np.array(corr_b)
        if b_corr_between:
            self.b_corr_between = np.array(b_corr_between)
        self.b_iter = b_iter
        self.b_samples = b_samples
        self.observed = observed
        self.rand_uncertainty = np.array(rand_uncertainty)
        if cov is None:
            self.invcov = None
        elif syst_uncertainty is not None:
            cov = np.ones(len(self.observed), len(self.observed)) * syst_uncertainty**2
            if rand_uncertainty is not None:
                cov += np.eye(len(self.observed)) * rand_uncertainty**2
            self.invcov = np.linalg.inv(np.ascontiguousarray(cov))
        else:
            self.invcov = np.linalg.inv(np.ascontiguousarray(cov))
        self.uplims = np.array(uplims)
        self.downlims = np.array(downlims)
        self.parallel_cores = parallel_cores
        self.ninput = n_input
        self.repeat_dims = np.array(repeat_dims)
        self.progress = progress
        if n_input:
            self.initial_guess = np.empty(n_input, dtype=object)
        elif hasattr(initial_guess, "__len__"):
            self.initial_guess = np.empty(len(initial_guess), dtype=object)
        else:
            self.initial_guess = np.array([initial_guess], dtype=float)

        if hasattr(initial_guess, "__len__"):
            if hasattr(initial_guess[0], "__len__"):
                self.initial_guess = np.empty(len(initial_guess), dtype=object)
                for i in range(len(initial_guess)):
                    self.initial_guess[i] = np.array(initial_guess[i], dtype=float)
            else:
                if n_input == len(initial_guess):
                    self.initial_guess = np.array(initial_guess, dtype=float)
                elif n_input == 1:
                    self.initial_guess = np.array(initial_guess, dtype=float)[None, :]
                else:
                    raise ValueError(
                        "your initial guess requires to specify the n_input quantity to indicate if these are multiple measurements for the same input qty (n_input=1) or a single measurement for different input_qty (n_input=len(initial_guess)). "
                    )
        self.circular = circular

    def measurement_function_x(self, theta):
        x = self.make_x_tuple(theta)
        if self.b is not None:
            xb = x + tuple(self.b)
        else:
            xb = x
        return self.measurement_function(*xb)

    def make_x_tuple(self, theta):
        x = copy.deepcopy(self.initial_guess)
        j = 0
        for i in range(len(x)):
            if not hasattr(x[i], "__len__"):
                x[i] = theta[j]
                j += 1
            else:
                for ii in range(len(x[i])):
                    if not hasattr(x[i][ii], "__len__"):
                        x[i][ii] = theta[j]
                        j += 1
                    else:
                        for iii in range(len(x[i][ii])):
                            if not hasattr(x[i][ii][iii], "__len__"):
                                x[i][ii][iii] = theta[j]
                                j += 1
                            else:
                                raise ValueError(
                                    "The initial guess has too high dimensionality."
                                )
        return tuple(x)

    def run_retrieval(
        self,
        nwalkers,
        steps,
        burn_in,
        return_samples=True,
        return_corr=False,
        include_b_results=False,
    ):
        """
        Function to run MCMC retrieval

        :param nwalkers: number of walkers (independent runs so that local minima can be avoided)
        :param steps: number of iterations each walkers should do
        :param burn_in: number of steps to be discarded at start because equilibrium has not yet been reached
        :param return_samples: boolean to indicate whether MCMC samples should be returned
        :param return_corr: boolean to indicate whether error correlation matrix should be returned
        :param include_b_results: boolean to indicate whether b-resuls should be returned
        :return: mean values for parameters, uncertainty in parameters, (optionally corr,)(optionally samples,)(optionally b_results,)
        """
        if hasattr(self.initial_guess, "__len__"):
            if hasattr(self.initial_guess[0], "__len__"):
                theta_0 = np.concatenate(self.initial_guess).flatten()
            else:
                theta_0 = np.array(self.initial_guess).flatten()
        else:
            theta_0 = np.array([self.initial_guess])

        if self.b is None:
            samples = self.run_MCMC(theta_0, nwalkers, steps, burn_in)
            b_samples = None
        else:
            prop = punpy.MCPropagation(self.b_iter)
            if self.b_samples is None:
                b_samples = prop.generate_MC_sample(
                    self.b, self.u_b, self.corr_b, self.b_corr_between
                )

            else:
                b_samples = self.b_samples

            if self.b_iter == 1:
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

        if self.circular:
            for samp in samples:
                if samp[0] < 0:
                    samp[0] = -samp[0]
                    samp[1] = samp[1] - 180
                if samp[1] < -180:
                    samp[1] += 360
                if samp[1] > 180:
                    samp[1] += -360

        return self.analyse_samples(
            samples, b_samples, return_samples, return_corr, include_b_results
        )

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

    def generate_theta_i(self, theta_0, factor_std=0.1):
        theta_i = theta_0 * np.random.normal(1.0, factor_std, theta_0.shape)
        if np.all(self.downlims.flatten() < theta_i) and np.all(
            self.uplims.flatten() > theta_i
        ):
            return theta_i
        else:
            # print(theta_i)
            return self.generate_theta_i(theta_0, factor_std=factor_std * 0.9)

    def analyse_samples(
        self, samples, b_samples, return_samples, return_corr, include_b_results
    ):
        if self.circular:
            for samp in samples:
                if samp[0] < 0:
                    samp[0] = -samp[0]
                    samp[1] = samp[1] - 180

                if samp[1] < -180:
                    samp[1] += 360

                if samp[1] > 180:
                    samp[1] += -360

        medians = np.median(samples, axis=0)
        unc_up = np.percentile(samples, 84, axis=0) - medians
        unc_down = -(np.percentile(samples, 16, axis=0) - medians)
        unc_avg = (unc_up + unc_down) / 2.0

        if return_corr:
            if samples.shape[1] > 1:
                corr = np.corrcoef(samples.T)
            else:
                corr = np.ones((1,))

        medians = self.make_x_tuple(medians)
        unc_avg = self.make_x_tuple(unc_avg)

        outs = (medians, unc_avg)
        if return_corr:
            outs += (corr,)

        if return_samples:
            outs += (samples,)

        if include_b_results:
            outs += (b_samples,)

        return outs

    def find_chisum(self, theta):
        model = self.measurement_function_x(theta)
        diff = model - self.observed
        if np.isfinite(np.sum(diff)):
            if self.invcov is None:
                return np.sum((diff) ** 2 / self.rand_uncertainty**2)
            else:
                # print(diff,np.linalg.inv(self.cov),np.dot(np.dot(diff.T,self.invcov),diff))
                if len(self.repeat_dims) == 0:
                    return np.dot(np.dot(diff.T, self.invcov), diff)
                elif len(self.repeat_dims) == 1:
                    sum = 0
                    for i in range(diff.shape[self.repeat_dims[0]]):
                        diffi = np.take(diff, i, self.repeat_dims[0])
                        sum += np.dot(np.dot(diffi.T, self.invcov), diffi)
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

    def lnlike(self, theta):
        # print(theta,self.find_chisum(theta))
        return -0.5 * (self.find_chisum(theta))

    def lnprior(self, theta):
        if np.all(self.downlims.flatten() < theta) and np.all(
            self.uplims.flatten() > theta
        ):
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
