"""Tests for retrieval_methods.LPU"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from curepy.retrieval_methods.LPU import LPU
from curepy.retrieval_methods.base import BaseRetrieval
from curepy.container.retrieval_input import RetrievalInput

class DummyRetrieval(BaseRetrieval):
    def run_retrieval(self, retrieval_inputs):
        self.retrieval_input = retrieval_inputs
        
class TestLPU(unittest.TestCase):

    def test_calculate_measurand_covariance_sy_and_sb_inv(self):
        lpu = LPU()
        J = np.eye(2)
        Sy_inv = 0.5 * np.eye(2)
        Sb_inv = 0.2 * np.eye(2)

        cov = lpu.calculate_measurand_covariance(
            None, J, Sy_inv, Sa_inv=None, Sb_inv=Sb_inv
        )

        expected = (1.0 / 0.7) * np.eye(2)
        np.testing.assert_allclose(cov, expected)

    def test_calculate_measurand_covariance_sa_inv(self):
        lpu = LPU()
        J = np.eye(2)
        Sy_inv = 0.5 * np.eye(2)
        Sb_inv = 0.2 * np.eye(2)
        Sa_inv = 0.2 * np.eye(2)

        cov = lpu.calculate_measurand_covariance(
            None, J, Sy_inv, Sa_inv=Sa_inv, Sb_inv=Sb_inv
        )

        expected = (1.0 / 0.9) * np.eye(2)
        np.testing.assert_allclose(cov, expected)
        
    def test_calculate_measurand_covariance_no_covariance(self):
        lpu = LPU()
        J = np.eye(2)
        Sy_inv = None
        Sb_inv = None

        with self.assertRaises(ValueError):
            cov = lpu.calculate_measurand_covariance(
                None, J, Sy_inv, Sa_inv=None, Sb_inv=Sb_inv
            )
    
    @patch.object(LPU, "calculate_Jb")
    def test_calculate_measurand_covariance_sy_inv(self, mock_calculate_Jb):
        lpu = LPU()
        J = np.eye(2)
        Sy_inv = 0.5 * np.eye(2)
        Sb_inv = None

        mock_calculate_Jb.return_value = np.eye(2)
        inp = RetrievalInput()
        inp.ancillary_obj = MagicMock()
        inp.ancillary_obj.calculate_b_cov = lambda: np.eye(2)
        lpu.retrieval_input = inp
        
        cov = lpu.calculate_measurand_covariance(
            None, J, Sy_inv, Sa_inv=None, Sb_inv=Sb_inv
        )

        expected = (3) * np.eye(2)
        np.testing.assert_allclose(cov, expected)

    @patch("curepy.retrieval_methods.LPU.LPU.calculate_measurand_covariance")
    @patch("comet_maths.convert_cov_to_corr")
    def test_process_inverse_jacobian(self, mock_convert, mock_calc_cov):
        lpu = LPU()
        lpu.retrieval_input = MagicMock()
        lpu.retrieval_input.measurement_obj.invcov = MagicMock(shape=(2, 2))

        covx = np.array([[4.0, 0.0], [0.0, 9.0]])
        mock_calc_cov.return_value = covx
        mock_convert.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])

        u_func, corr_x = lpu.process_inverse_jacobian(None, np.array([0.0, 0.0]))
        np.testing.assert_allclose(u_func, np.array([2.0, 3.0]))
        np.testing.assert_array_equal(corr_x, np.array([[1.0, 0.0], [0.0, 1.0]]))

    @patch("comet_maths.calculate_Jacobian")
    def test_calculate_Jx_calls_calculate_Jacobian(self, mock_calc_jac):
        lpu = LPU()
        lpu.retrieval_input = MagicMock()
        lpu.retrieval_input.measurement_function_obj.measurement_function_flattened_output = (
            MagicMock()
        )
        lpu.retrieval_input.ancillary_obj.b = MagicMock()

        mock_calc_jac.return_value = np.array([[1.0]])
        Jx = lpu.calculate_Jx(np.array([1.0]))
        np.testing.assert_array_equal(Jx, np.array([[1.0]]))
        self.assertEqual(mock_calc_jac.call_count, 1)


if __name__ == "__main__":
    unittest.main()
