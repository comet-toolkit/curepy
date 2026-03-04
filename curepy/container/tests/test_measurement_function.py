"""Tests for MeasurementFunction class"""

import unittest
import numpy as np
import numpy.testing as npt
from unittest.mock import patch

from curepy.container.measurement_function import MeasurementFunction

def dummy_function(x, b):
    return x + b
    
class TestMeasurementFunction(unittest.TestCase):
    def test_init_builds_basic(self):
        mf = MeasurementFunction(dummy_function,
                                 np.array([5, 10]))
        
        out = mf.measurement_function(mf.initial_guess, 
                                      6)
        npt.assert_array_equal(out, np.array([11,16]))