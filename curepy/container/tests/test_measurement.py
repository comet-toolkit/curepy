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
        
    def test__format_correlation_none(self):
        corr = None
        formatted_corr = Measurement._format_correlation(y, corr)
        
        self.assertIsNone(formatted_corr)
        
    def test__format_correlation_rand(self):
        corr = "rand"
        formatted_corr = Measurement._format_correlation(y, corr)
        
        np.testing.assert_array_equal(formatted_corr, np.diag(np.diag(formatted_corr)))
        
    def test__format_correlation_syst(self):
        corr = "syst"
        formatted_corr = Measurement._format_correlation(y, corr)
        
        np.testing.assert_array_equal(formatted_corr, np.ones((len(y), len(y))))
        
    def test__format_correlation_custom(self):
        corr = np.ones((len(y), len(y))) + np.eye(len(y))
        formatted_corr = Measurement._format_correlation(y, corr)
        
        np.testing.assert_array_equal(formatted_corr, corr)
        
    @patch("comet_maths.convert_corr_to_cov")
    def test_calculate_inv_cov_diag(self, mock_convert_corr_to_cov):
           mock_convert_corr_to_cov.return_value = 0.5*np.eye(5)
           
           invcov = Measurement.calculate_inv_cov(y, np.ones_like((len(y), len(y))))
           
           np.testing.assert_array_equal(invcov, 2*np.eye(5))
        
    @patch("comet_maths.convert_corr_to_cov")
    @patch("numpy.linalg.inv")
    def test_calculate_inv_cov(self, mock_convert_corr_to_cov, mock_inv):
           mock_convert_corr_to_cov.return_value = 0.5*np.ones((5,5))
           
           invcov = Measurement.calculate_inv_cov(y, np.ones_like((len(y), len(y))))
           
           self.assertEqual(mock_inv.call_count, 1)
    
    #add tests for check shapes when finalised
        
        
if __name__ == "__main__":
    unittest.main()