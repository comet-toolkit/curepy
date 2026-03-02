"""Tests for retrieval_methods.MCMC"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from curepy.retrieval_methods.MCMC import MCMC


class DummyPrior:
    def lnprior(self, theta):
        # Return a function that when called returns finite array
        return lambda: np.array([0.0 for _ in np.atleast_1d(theta)])


class TestMCMC(unittest.TestCase):

    def test_generate_theta_i_shape_and_prior_check(self):
        m = MCMC(nwalkers=2, steps=10, burn_in=1)
        # set retrieval_input and prior
        m.retrieval_input = type("R", (), {})()
        m.retrieval_input.prior_obj = DummyPrior()

        theta_0 = np.array([1.0, 2.0])
        ti = m.generate_theta_i(theta_0, factor_std=0.01)
        self.assertEqual(ti.shape, theta_0.shape)

    def test_analyse_samples_basic(self):
        m = MCMC(nwalkers=2, steps=10, burn_in=1)
        # create simple samples: 3 samples, 1-dim
        samples = np.array([[0.0], [1.0], [2.0]])
        res = m.analyse_samples(samples, b_samples=None, return_samples=True, return_corr=True, return_b_samples=False, reshape_results=False)

        # medians should be 1.0, uncertainties average ((84-50)+(50-16))/2 ~ 1.0
        self.assertEqual(res.values.tolist(), [1.0])
        self.assertEqual(res.samples.tolist(), samples.tolist())
        # for 1-dim corr should be ones((1,)) per implementation
        self.assertTrue(np.allclose(res.correlation, np.ones((1,))))

    @patch("curepy.retrieval_methods.MCMC.emcee.EnsembleSampler")
    def test_run_MCMC_uses_sampler(self, mock_sampler_class):
        m = MCMC(nwalkers=2, steps=5, burn_in=0)
        theta_0 = np.array([1.0])
        # fake sampler instance
        fake_sampler = MagicMock()
        # make get_chain return shaped array [nwalkers, steps, ndim]
        fake_chain = np.zeros((2, 5, 1)) + 2.0
        fake_sampler.run_mcmc = MagicMock()
        fake_sampler.get_chain = MagicMock(return_value=fake_chain)
        mock_sampler_class.return_value = fake_sampler

        samples = m.run_MCMC(theta_0, m.nwalkers, m.steps, m.burn_in)
        # our fake chain contains only value 2.0 so samples should contain 2.0
        self.assertTrue(np.all(samples == 2.0))


if __name__ == "__main__":
    unittest.main()
