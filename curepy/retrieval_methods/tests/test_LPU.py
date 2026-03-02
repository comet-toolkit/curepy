"""Tests for retrieval_methods.LPU"""

import unittest
import numpy as np
from unittest.mock import patch

from curepy.retrieval_methods.LPU import LPU


class TestLPU(unittest.TestCase):

    def test_calculate_measurand_covariance_sy_and_sb_inv(self):
        lpu = LPU()
        J = np.eye(2)
        Sy_inv = 0.5 * np.eye(2)
        Sb_inv = 0.2 * np.eye(2)

        # when Sa_inv is None
        cov = lpu.calculate_measurand_covariance(None, J, Sy_inv, Sa_inv=None, Sb_inv=Sb_inv)
        # Se_inv = Sy_inv + Sb_inv = 0.7 I, then inv(J.T Se_inv J) = inv(0.7 I) = (1/0.7) I
        expected = (1.0 / 0.7) * np.eye(2)
        np.testing.assert_allclose(cov, expected)

    @patch("curepy.retrieval_methods.LPU.calculate_measurand_covariance")
    @patch("curepy.retrieval_methods.LPU.cm.convert_cov_to_corr")
    def test_process_inverse_jacobian(self, mock_convert, mock_calc_cov):
        lpu = LPU()
        # setup retrieval_input placeholder
        lpu.retrieval_input = type("R", (), {})()
        lpu.retrieval_input.prior_obj = None

        covx = np.array([[4.0, 0.0], [0.0, 9.0]])
        mock_calc_cov.return_value = covx
        mock_convert.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])

        u_func, corr_x = lpu.process_inverse_jacobian(None, np.array([0.0, 0.0]))
        np.testing.assert_allclose(u_func, np.array([2.0, 3.0]))
        np.testing.assert_array_equal(corr_x, np.array([[1.0, 0.0], [0.0, 1.0]]))

    @patch("curepy.retrieval_methods.LPU.cm.calculate_Jacobian")
    def test_calculate_Jx_calls_calculate_Jacobian(self, mock_calc_jac):
        lpu = LPU()
        # build minimal retrieval_input
        class MF:
            def measurement_function_flattened_output(self, x, b=None):
                return np.array([x[0] + (b[0] if b is not None else 0)])

        class Anc:
            b = [np.array(1.0)]

        lpu.retrieval_input = type("R", (), {"measurement_function_obj": MF(), "ancillary_obj": Anc()})()

        mock_calc_jac.return_value = np.array([[1.0]])
        Jx = lpu.calculate_Jx(np.array([1.0]))
        np.testing.assert_array_equal(Jx, np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main()
