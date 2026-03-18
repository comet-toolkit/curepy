"""Tests for retrieval_methods.base"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from curepy.container.retrieval_input import RetrievalInput
from curepy.retrieval_methods.base import BaseRetrieval


def make_mock_retrieval_input_for_chisum(
    measurement_function_output,
    y_flat,
    invcov=None,
    u_y=None,
    b=None,
):
    retrieval_input = RetrievalInput()
    # measurement function object
    retrieval_input.measurement_function_obj = MagicMock()
    retrieval_input.measurement_function_obj.measurement_function_x = MagicMock(
        return_value=measurement_function_output
    )

    # measurement object
    retrieval_input.measurement_obj = MagicMock()
    retrieval_input.measurement_obj.y_flat = np.array(y_flat)
    retrieval_input.measurement_obj.invcov = invcov
    retrieval_input.measurement_obj.u_y = u_y
    # ancillary
    retrieval_input.ancillary_obj = MagicMock()
    retrieval_input.ancillary_obj.b = b

    return retrieval_input


class DummyRetrieval(BaseRetrieval):
    def _run_retrieval(self, retrieval_inputs):
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
        TEST_INPUT.build_prior(["uniform"], [{"minimum": 0, "maximum": 1}])
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
        self.assertEqual(
            dr.retrieval_input.prior_obj.prior_params,
            [{"minimum": -np.inf, "maximum": np.inf}],
        )

    def test__check_retrieval_input_no_ancill(self):
        dr = DummyRetrieval()
        TEST_INPUT = RetrievalInput()
        TEST_INPUT.build_prior(["uniform"], [{"minimum": 0, "maximum": 1}])
        dr.run_retrieval(TEST_INPUT)
        dr._check_retrieval_input()
        assert dr.retrieval_input.ancillary_obj

    def test_chisum_no_invcov(self):
        measurement_output = np.array([1.0, 2.0, 3.0])
        y_flat = np.array([1.0, 1.0, 1.0])
        u_y = np.array([1.0, 1.0, 1.0])

        retrieval_input = make_mock_retrieval_input_for_chisum(
            measurement_function_output=measurement_output,
            y_flat=y_flat,
            invcov=None,
            u_y=u_y,
            b=None,
        )

        dr = DummyRetrieval()
        dr.run_retrieval(retrieval_input)

        result = dr.find_chisum(theta=None)
        expected = np.sum(((measurement_output - y_flat) ** 2) / (u_y**2))
        assert result == expected

    def test_multiple_repeat_dims_raises_error(self):
        retrieval_input = make_mock_retrieval_input_for_chisum(
            measurement_function_output=np.array([1.0, 2.0]),
            y_flat=np.array([1.0, 1.0]),
            invcov=np.eye(2),
            u_y=None,
            b=None,
        )

        dr = DummyRetrieval()
        dr.run_retrieval(retrieval_input)
        with self.assertRaises(ValueError):
            dr.find_chisum(theta=None, repeat_dims=[0, 1])

    def test_returns_inf_when_diff_contains_nan(self):
        """If diff contains non-finite values, should return np.inf."""

        measurement_output = np.array([np.nan, 1.0])
        y_flat = np.array([1.0, 1.0])
        u_y = np.array([1.0, 1.0])

        retrieval_input = make_mock_retrieval_input_for_chisum(
            measurement_function_output=measurement_output,
            y_flat=y_flat,
            invcov=None,
            u_y=u_y,
            b=None,
        )

        dr = DummyRetrieval()
        dr.run_retrieval(retrieval_input)

        result = dr.find_chisum(theta=None)
        assert np.isinf(result)

    def test_chisum_with_invcov_no_repeat(self):
        """Test dot(diff.T, invcov, diff) when invcov exists."""

        measurement_output = np.array([2.0, 4.0])
        y_flat = np.array([1.0, 1.0])
        invcov = np.eye(2)

        retrieval_input = make_mock_retrieval_input_for_chisum(
            measurement_function_output=measurement_output,
            y_flat=y_flat,
            invcov=invcov,
            u_y=None,
            b=None,
        )

        dr = DummyRetrieval()
        dr.run_retrieval(retrieval_input)

        result = dr.find_chisum(theta=None)

        diff = measurement_output - y_flat
        expected = diff.T @ invcov @ diff

        assert result == expected

    @patch.object(BaseRetrieval, "find_chisum")
    def test_lnprob_valid(self, mock_find_chisum):
        dr = DummyRetrieval()
        TEST_INPUT = RetrievalInput()
        TEST_INPUT.prior_obj = MagicMock()
        TEST_INPUT.prior_obj.lnprior = lambda theta: (lambda: np.array([0.0]))
        mock_find_chisum.return_value = 5
        dr.run_retrieval(TEST_INPUT)
        out = dr.lnprob([3])
        self.assertEqual(out, -2.5)

    @patch.object(BaseRetrieval, "find_chisum")
    def test_lnprob_invalid(self, mock_find_chisum):
        dr = DummyRetrieval()
        TEST_INPUT = RetrievalInput()
        TEST_INPUT.prior_obj = MagicMock()
        TEST_INPUT.prior_obj.lnprior = lambda theta: (lambda: np.array([-np.inf]))
        mock_find_chisum.return_value = 5
        dr.run_retrieval(TEST_INPUT)
        out = dr.lnprob([3])
        self.assertEqual(out, -np.inf)


if __name__ == "__main__":
    unittest.main()
