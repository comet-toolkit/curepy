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
        meas = Measurement(y, u_y, corr)
        
        self.assertIsNone(meas.corr_y)
        
    def test__format_correlation_rand(self):
        corr = "rand"
        meas = Measurement(y, u_y, corr)
        
        np.testing.assert_array_equal(meas.corr_y, np.diag(np.diag(meas.corr_y)))
        
    @patch.object(Measurement, "calculate_inv_cov")
    def test__format_correlation_syst(self, mock_calculate_inv_cov):
        corr = "syst"
        meas = Measurement(y, u_y, corr)
        
        np.testing.assert_array_equal(meas.corr_y, np.ones((len(y), len(y))))
        
    @patch.object(Measurement, "calculate_inv_cov")
    def test__format_correlation_custom(self, mock_calculate_inv_cov):
        corr = np.ones((len(y), len(y))) + np.eye(len(y))
        meas = Measurement(y, u_y, corr)
        
        np.testing.assert_array_equal(meas.corr_y, corr)
        
        
        
        
        
        
if __name__ == "__main__":
    unittest.main()