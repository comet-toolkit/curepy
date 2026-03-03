"""Tests for retrieval_methods.base"""

import unittest
import numpy as np
from unittest.mock import MagicMock

from curepy.container.retrieval_input import RetrievalInput
from curepy.retrieval_methods.base import BaseRetrieval

class DummyRetrieval(BaseRetrieval):
    def run_retrieval(self, retrieval_inputs):
        self.retrieval_input = retrieval_inputs


class TestBaseRetrieval(unittest.TestCase):

    def test_generate_theta_0_scalar(self):
        t = BaseRetrieval.generate_theta_0(3.0)
        np.testing.assert_array_equal(t, np.array([3.0]))

    def test_generate_theta_0_list(self):
        t = BaseRetrieval.generate_theta_0([1.0, 2.0])
        np.testing.assert_array_equal(t, np.array([1.0, 2.0]))

    def test_generate_theta_0_nested(self):
        t = BaseRetrieval.generate_theta_0([[1.0], [2.0]])
        np.testing.assert_array_equal(t, np.array([1.0, 2.0]))

    def test_reshape_outputs(self):
        dr = DummyRetrieval()

        dr.retrieval_input = MagicMock()
        dr.retrieval_input.measurement_function_obj.initial_guess = MagicMock(
            shape=(2, 2)
        )

        x = np.arange(4)
        u_x = np.arange(4).astype(float)
        corr = None

        x_r, u_r, c_r = dr.reshape_outputs(x, u_x, corr)
        self.assertEqual(x_r.shape, (2, 2))
        self.assertEqual(u_r.shape, (2, 2))

    def test_reshape_outputs_incorrect_shape(self):
        dr = DummyRetrieval()

        dr.retrieval_input = MagicMock()
        dr.retrieval_input.measurement_function_obj.initial_guess = MagicMock(
            shape=(2, 2)
        )

        x = np.arange(3)
        u_x = np.arange(4).astype(float)
        corr = None

        with self.assertRaises(ValueError):
            x_r, u_r, c_r = dr.reshape_outputs(x, u_x, corr)
    
    def test__check_retrieval_input_good(self):
        dr = DummyRetrieval()
        TEST_INPUT = RetrievalInput()
        TEST_INPUT.build_prior(['uniform'], [{'minimum': 0, 'maximum': 1}])
        TEST_INPUT.build_ancillary()
        dr.run_retrieval(TEST_INPUT)
        dr._check_retrieval_input()
        
    def test__check_retrieval_input_no_prior(self):
        dr = DummyRetrieval()
        TEST_INPUT = RetrievalInput()
        TEST_INPUT.build_ancillary()
        dr.run_retrieval(TEST_INPUT)
        dr.retrieval_input.measurement_function_obj = MagicMock()
        dr.retrieval_input.measurement_function_obj.initial_guess = [0]
        dr._check_retrieval_input()
        self.assertEqual(dr.retrieval_input.prior_obj.prior_params, [{'minimum': -np.inf, 'maximum': np.inf}])
        
    def test__check_retrieval_input_no_ancill(self):
        dr = DummyRetrieval()
        TEST_INPUT = RetrievalInput()
        TEST_INPUT.build_prior(['uniform'], [{'minimum': 0, 'maximum': 1}])
        dr.run_retrieval(TEST_INPUT)
        dr._check_retrieval_input()
        assert(dr.retrieval_input.ancillary_obj)
        


if __name__ == "__main__":
    unittest.main()
