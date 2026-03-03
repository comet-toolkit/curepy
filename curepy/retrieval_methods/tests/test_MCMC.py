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
        m.retrieval_input = MagicMock()
        m.retrieval_input.prior_obj = DummyPrior()

        theta_0 = np.array([1.0, 2.0])
        ti = m.generate_theta_i(theta_0, factor_std=0.01)
        self.assertEqual(ti.shape, theta_0.shape)

    def test_analyse_samples_basic(self):
        m = MCMC(nwalkers=2, steps=10, burn_in=1)
        # create simple samples: 3 samples, 1-dim
        samples = np.array([[0.0], [1.0], [2.0]])
        res = m.analyse_samples(
            samples,
            b_samples=None,
            return_samples=True,
            return_corr=True,
            return_b_samples=False,
            reshape_results=False,
        )

        # medians should be 1.0, uncertainties average ((84-50)+(50-16))/2 ~ 1.0
        self.assertEqual(res.values.tolist(), [1.0])
        self.assertEqual(res.samples.tolist(), samples.tolist())
        # for 1-dim corr should be ones((1,)) per implementation
        self.assertTrue(np.allclose(res.correlation, np.ones((1,))))

    @patch("curepy.retrieval_methods.MCMC.emcee.EnsembleSampler")
    @patch.object(MCMC, "generate_theta_i")
    def test_run_MCMC_uses_sampler(self, mock_generate, mock_sampler_class):
        m = MCMC(nwalkers=2, steps=5, burn_in=0)
        theta_0 = np.array([1.0])
        mock_generate.return_value = [0 for i in range(2)]
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

    def test_run_retrieval_without_b_samples(self):
        # test branch where ancillary.b_samples is None and run_MCMC is called once
        m = MCMC(nwalkers=2, steps=4, burn_in=1)

        # build retrieval_input mock
        retrieval_input = MagicMock()
        retrieval_input.measurement_function_obj = MagicMock()
        retrieval_input.measurement_function_obj.initial_guess = np.array([1.0])
        # ancillary that generates no b_samples
        anc = MagicMock()

        def gen_b():
            anc.b_samples = None
            anc.b_MC_steps = 1

        anc.generate_b_samples = gen_b
        anc.b = [np.array(0.0)]
        retrieval_input.ancillary_obj = anc

        # attach minimal prior
        retrieval_input.prior_obj = MagicMock()
        retrieval_input.prior_obj.lnprior = lambda theta: (lambda: np.array([0.0]))

        m.retrieval_input = retrieval_input

        # patch run_MCMC to return predictable samples
        samples_returned = np.ones(((m.nwalkers * m.steps - m.burn_in), 1)) * 3.0
        m.run_MCMC = MagicMock(return_value=samples_returned)

        result = m.run_retrieval(retrieval_input, return_samples=True)
        # result.samples should be the same samples returned by our mocked run_MCMC
        self.assertTrue(np.allclose(result.samples, samples_returned))

    def test_run_retrieval_with_b_samples_combination(self):
        # test branch where ancillary.b_samples contains multiple sets and results are concatenated
        m = MCMC(nwalkers=2, steps=3, burn_in=1)

        retrieval_input = MagicMock()
        retrieval_input.measurement_function_obj = MagicMock()
        retrieval_input.measurement_function_obj.initial_guess = np.array([1.0])

        # create ancillary b_samples as list-of-arrays: two parameters, two samples each
        anc = MagicMock()
        anc.b = [np.array(0.0), np.array(0.0)]
        # b_samples shaped as [param][sample_index]
        anc.b_samples = [np.array([10.0, 20.0]), np.array([30.0, 40.0])]
        anc.b_MC_steps = 2

        def gen_b():
            # already set
            return None

        anc.generate_b_samples = gen_b
        retrieval_input.ancillary_obj = anc

        retrieval_input.prior_obj = MagicMock()
        retrieval_input.prior_obj.lnprior = lambda theta: (lambda: np.array([0.0]))

        m.retrieval_input = retrieval_input

        # make run_MCMC return distinct arrays so concatenation can be tested
        single_return = np.ones(((m.nwalkers * m.steps - m.burn_in), 1)) * 5.0
        m.run_MCMC = MagicMock(return_value=single_return)

        result = m.run_retrieval(retrieval_input, return_samples=True)
        # total samples should be b_samples[0].size * single_return.shape[0]
        expected_len = anc.b_samples[0].size * single_return.shape[0]
        self.assertEqual(result.samples.shape[0], expected_len)

    def test_generate_theta_i_recurses_until_prior_accepts(self):
        m = MCMC(nwalkers=2, steps=10, burn_in=1)

        # create prior that rejects once then accepts
        class FlakyPrior:
            def __init__(self):
                self.calls = 0

            def lnprior(self, theta):
                self.calls += 1
                if self.calls == 1:
                    return lambda: np.array([-np.inf])
                else:
                    return lambda: np.array([0.0 for _ in np.atleast_1d(theta)])

        m.retrieval_input = MagicMock()
        m.retrieval_input.prior_obj = FlakyPrior()

        theta_0 = np.array([1.0, 2.0])
        # to make recursion deterministic, patch random.normal to slightly vary
        original_rand = np.random.normal
        np.random.normal = lambda loc, scale, size: np.ones(size)
        try:
            ti = m.generate_theta_i(theta_0, factor_std=0.01)
        finally:
            np.random.normal = original_rand

        self.assertEqual(ti.shape, theta_0.shape)

    @patch("curepy.retrieval_methods.MCMC.Pool")
    @patch("curepy.retrieval_methods.MCMC.emcee.EnsembleSampler")
    @patch.object(MCMC, "generate_theta_i")
    def test_run_MCMC_parallel_uses_pool(
        self, mock_generate, mock_sampler_class, mock_pool_class
    ):
        m = MCMC(nwalkers=2, steps=4, burn_in=0, parallel_cores=2)
        theta_0 = np.array([1.0])
        mock_generate.return_value = [0 for i in range(2)]
        fake_sampler = MagicMock()
        fake_chain = np.zeros((2, 4, 1)) + 7.0
        fake_sampler.run_mcmc = MagicMock()
        fake_sampler.get_chain = MagicMock(return_value=fake_chain)
        mock_sampler_class.return_value = fake_sampler

        # call run_MCMC which should instantiate Pool and pass it to EnsembleSampler
        samples = m.run_MCMC(theta_0, m.nwalkers, m.steps, m.burn_in)
        mock_pool_class.assert_called_once()


if __name__ == "__main__":
    unittest.main()
