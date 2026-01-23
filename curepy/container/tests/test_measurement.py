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
        
    @patch.object(Measurement, "calculate_inv_cov")
    def test__format_correlation_syst(self, mock_calculate_inv_cov):
        corr = "syst"
        formatted_corr = Measurement._format_correlation(y, corr)
        
        np.testing.assert_array_equal(formatted_corr, np.ones((len(y), len(y))))
        
    def test__format_correlation_custom(self):
        corr = np.ones((len(y), len(y))) + np.eye(len(y))
        formatted_corr = Measurement._format_correlation(y, corr)
        
        np.testing.assert_array_equal(formatted_corr, corr)
        
        
        
        
        
        
if __name__ == "__main__":
    unittest.main()