"""Tests for Measurement container class"""

import unittest
import numpy as np
from unittest.mock import patch
from curepy.container.measurement import Measurement

y = np.linspace(0, 100, 47)
u_y = 0.04 * y


class TestMeasurement(unittest.TestCase):

    def test___init__no_corr_none_invcov(self):

        meas = Measurement(y, u_y)

        self.assertIsNone(meas.invcov)

    @patch("comet_maths.convert_corr_to_cov")
    def test_calculate_inv_cov_diag(self, mock_convert_corr_to_cov):
        mock_convert_corr_to_cov.return_value = 0.5 * np.eye(5)

        invcov = Measurement.calculate_inv_cov(y, np.ones_like((len(y), len(y))))

        np.testing.assert_array_equal(invcov, 2 * np.eye(5))

    @patch("comet_maths.convert_corr_to_cov")
    @patch("numpy.linalg.inv")
    def test_calculate_inv_cov(self, mock_convert_corr_to_cov, mock_inv):
        mock_convert_corr_to_cov.return_value = 0.5 * np.ones((5, 5))

        invcov = Measurement.calculate_inv_cov(y, np.ones_like((len(y), len(y))))

        self.assertEqual(mock_inv.call_count, 1)

    @patch("comet_maths.convert_corr_to_cov")
    @patch.object(Measurement, "_check_shapes")
    @patch("curepy.utilities.utilities.format_correlation")
    def test_init_format_correlation_called(self, mock_format, mock_check, mock_convert):
        
        meas = Measurement(y, u_y, corr_y = 'rand')

        self.assertEqual(mock_format.call_count, 1)

    def test__flatten_inputs_y(self):
        y = np.ones((2,2))
        y_flat, u_y_flat, y_shape = Measurement._flatten_inputs(y, None)
        
        self.assertEqual(y_flat.shape, (4,))
        self.assertIsNone(u_y_flat)
        self.assertEqual(y.shape, y_shape)
        
    def test__flatten_inputs_invalid_shapes(self):
        y = np.ones((2,2))
        u_y = np.ones((3,2))
        with self.assertRaises(ValueError):
            y_flat, u_y_flat, y_shape = Measurement._flatten_inputs(y, u_y)
        
    def test__flatten_inputs_u_y(self):
        y = np.ones((2,2))
        u_y = 0.1 * np.ones((2,2))
        y_flat, u_y_flat, y_shape = Measurement._flatten_inputs(y, u_y)
        
        self.assertEqual(y_flat.shape, u_y_flat.shape)
        self.assertEqual(u_y_flat.shape, (4,))
           
    def test__check_shapes_pass(self):
        corr_y = np.eye(len(y))
        Measurement._check_shapes(y, u_y, corr_y)
    
    def test__check_shapes_mismatch_y_u(self):
        y = np.linspace(0, 1, 4)
        with self.assertRaises(ValueError):
            Measurement._check_shapes(y, u_y, None)

    def test__check_shapes_no_u(self):
        corr_y = np.eye(len(y))
        with self.assertRaises(ValueError):
            Measurement._check_shapes(y, None, corr_y)
    
    def test__check_shapes_corr_rectangle(self):
        corr_y = np.ones((len(y), len(y) + 1))
        with self.assertRaises(ValueError):
            Measurement._check_shapes(y, u_y, corr_y)
    
    def test__check_shapes_mismatch_y_corr(self):
        corr_y = np.eye(len(y) + 1)
        with self.assertRaises(ValueError):
            Measurement._check_shapes(y, u_y, corr_y)
    
    
        
        
        
if __name__ == "__main__":
    unittest.main()
