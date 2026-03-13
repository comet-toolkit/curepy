"""Markov Chain Monte Carlo (MCMC) retrieval class"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.container.retrieval_input import RetrievalInput
from curepy.container.retrieval_result import RetrievalResult


from multiprocessing import Pool
import emcee
import numpy as np
from typing import Optional


class MCMC(BaseRetrieval):
    """MCMC retrieval object."""

    def __init__(
        self,
        nwalkers: int,
        steps: int,
        burn_in: int,
        progress: bool = True,
        parallel_cores: int = 1,
    ) -> None:
        """
        Initialise the MCMC retrieval object.

        :param nwalkers: Number of ensemble walkers used by
            :class:`emcee.EnsembleSampler`.
        :param steps: Total number of MCMC steps per walker.
        :param burn_in: Number of initial samples to discard as burn-in.
        :param progress: If ``True``, display a progress bar during sampling.
        :param parallel_cores: Number of CPU cores for parallel sampling.
            Values greater than 1 use :class:`multiprocessing.Pool`.
        """

        self.nwalkers = nwalkers
        self.steps = steps
        self.burn_in = burn_in

        self.progress = progress
        self.parallel_cores = parallel_cores

    def run_retrieval(
        self,
        retrieval_input: RetrievalInput,
        return_samples: bool = False,
        return_corr: bool = False,
        return_b_samples: bool = False,
        reshape_results: bool = True,
    ) -> RetrievalResult:
        """
        Run the MCMC retrieval and return the results.

        :param retrieval_input: Object containing all retrieval inputs.
        :param return_samples: If ``True``, the full sample array is stored
            in the returned :class:`~curepy.container.retrieval_result.RetrievalResult`.
        :param return_corr: If ``True``, the parameter correlation matrix is
            computed from the samples and stored in the result.
        :param return_b_samples: If ``True``, the ancillary parameter samples
            are stored in the result.
        :param reshape_results: If ``True``, reshape the flat output arrays
            to the initial-guess shape.
        :returns: Object containing retrieved values, uncertainties, and
            optionally samples and correlations.
        """

        self.retrieval_input = retrieval_input

        self._check_retrieval_input()

        # define theta_0
        theta_0 = self.generate_theta_0(
            self.retrieval_input.measurement_function_obj.initial_guess
        )

        # generate b samples if ancillary data exists
        self.retrieval_input.ancillary_obj.generate_b_samples()
            
        b_samples = self.retrieval_input.ancillary_obj.b_samples

        # generate samples with MCMC
        if b_samples is None or self.retrieval_input.ancillary_obj.b_MC_steps == 1:
            samples = self.run_MCMC(theta_0, self.nwalkers, self.steps, self.burn_in)
        else:
            samples = np.zeros(
                (
                    (self.nwalkers * self.steps - self.burn_in)
                    * self.retrieval_input.ancillary_obj.b_MC_steps,
                    len(theta_0),
                ),
                dtype=np.float32,
            )

            b = self.retrieval_input.ancillary_obj.b[:]

            for i in range(len(b_samples[0])):
                for ii in range(len(b_samples)):
                    # if b_samples[ii].ndim == 1:
                    self.retrieval_input.ancillary_obj.b[ii] = b_samples[ii][i]
                    # elif b_samples[ii].ndim == 2:
                    #     self.retrieval_input.ancillary_obj.b[ii] = np.array(
                    #         [b_samples[ii][j][i] for j in range(len(b_samples[ii]))]
                    #     )
                    # else:
                    #     raise ValueError(
                    #         "MCMC_retrieval: the dimensionality of one of the parameters in b is not supported (currently the ancillary parameters in b can only be floats or 1d arrays)."
                    #     )

                samples[
                    i
                    * (self.nwalkers * self.steps - self.burn_in) : (i + 1)
                    * (self.nwalkers * self.steps - self.burn_in),
                    :,
                ] = self.run_MCMC(theta_0, self.nwalkers, self.steps, self.burn_in)

            self.retrieval_input.ancillary_obj.b = b[:]

        result = self.analyse_samples(
            samples,
            b_samples,
            return_samples,
            return_corr,
            return_b_samples,
            reshape_results,
        )
        
        result.calculate_statistics(self.retrieval_input)
        
        return result

    def run_MCMC(
        self,
        theta_0: np.ndarray,
        nwalkers: int,
        steps: int,
        burn_in: int,
    ) -> np.ndarray:
        """
        Run :class:`emcee.EnsembleSampler` and return the post-burn-in chain.

        :param theta_0: Initial state vector around which walkers are
            initialised.
        :param nwalkers: Number of ensemble walkers.
        :param steps: Total number of sampling steps.
        :param burn_in: Number of initial samples to discard.
        :returns: Array of post-burn-in samples with shape
            ``(nwalkers * steps - burn_in, ndim)``.
        """
        ndimw = len(theta_0)
        pos = [self.generate_theta_i(theta_0) for i in range(nwalkers)]

        if self.parallel_cores > 1:
            p = Pool(self.parallel_cores)
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob, pool=p)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndimw, self.lnprob)
        sampler.run_mcmc(pos, steps, progress=self.progress)

        samples = sampler.get_chain()[:, :, :].reshape((-1, ndimw))[burn_in::]
        return samples

    def generate_theta_i(
        self,
        theta_0: np.ndarray,
        factor_std: float = 0.1,
    ) -> np.ndarray:
        """
        Generate a single walker starting position from ``theta_0``.

        Perturbs ``theta_0`` by a Gaussian factor and recursively reduces
        the perturbation magnitude until the resulting position lies within
        the support of the prior.

        :param theta_0: Initial state vector.
        :param factor_std: Standard deviation of the multiplicative Gaussian
            perturbation.
        :returns: Perturbed starting position that is within the prior
            support.
        """
        theta_i = theta_0 * np.random.normal(1.0, factor_std, theta_0.shape)
        if all(
            np.isfinite(
                self.retrieval_input.prior_obj.lnprior(
                    theta_i,
                )()
            )
        ):
            return theta_i
        else:
            return self.generate_theta_i(theta_0, factor_std=factor_std * 0.9)

    def analyse_samples(
        self,
        samples: np.ndarray,
        b_samples: Optional[np.ndarray],
        return_samples: bool,
        return_corr: bool,
        return_b_samples: bool,
        reshape_results: bool,
    ) -> RetrievalResult:
        """
        Summarise MCMC samples into a :class:`~curepy.container.retrieval_result.RetrievalResult`.

        Computes the median, symmetric uncertainty (average of upper and
        lower 1-sigma percentiles), and optionally the correlation matrix.

        :param samples: Post-burn-in MCMC samples.
        :param b_samples: Ancillary parameter samples.
        :param return_samples: If ``True``, include the raw samples in the
            result.
        :param return_corr: If ``True``, compute and include the correlation
            matrix.
        :param return_b_samples: If ``True``, include ancillary samples.
        :param reshape_results: If ``True``, reshape outputs to the
            initial-guess shape.
        :returns: Retrieved values, uncertainties, and optional extras.
        """

        medians = np.median(samples, axis=0)
        unc_up = np.percentile(samples, 84, axis=0) - medians
        unc_down = -(np.percentile(samples, 16, axis=0) - medians)
        unc_avg = (unc_up + unc_down) / 2.0

        if return_corr:
            if samples.shape[1] > 1:
                corr = np.corrcoef(samples.T)
            else:
                corr = np.ones((1,))

        if reshape_results:
            medians, unc_avg, corr = self.reshape_outputs(
                medians, unc_avg, corr if return_corr else None
            )

        outs = RetrievalResult(
            x=medians,
            u_x=unc_avg,
            corr_x=corr if return_corr else None,
            samples=samples if return_samples else None,
            b_samples=b_samples if return_b_samples else None,
            x_names=self.retrieval_input.measurement_function_obj._input_quantities_names,
        )

        return outs
