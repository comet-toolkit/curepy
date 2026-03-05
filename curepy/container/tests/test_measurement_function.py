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
        
    def test__format_initial_guess_scalar(self):
        initial_guess = 5
        formatted_guess = MeasurementFunction._format_initial_guess(initial_guess)
        npt.assert_array_equal(formatted_guess, np.array([5]))
        
    def test__format_initial_guess_list(self):
        initial_guess = [5, 6, 8]
        formatted_guess = MeasurementFunction._format_initial_guess(initial_guess)
        npt.assert_array_equal(formatted_guess, np.array([5, 6, 8]))
        
    def test__format_initial_guess_array(self):
        initial_guess = np.ones((3, 2))
        formatted_guess = MeasurementFunction._format_initial_guess(initial_guess)
        npt.assert_array_equal(formatted_guess, np.array([[1,1], [1,1], [1,1]]))
        
    def test__format_initial_guess_ragged_array(self):
        initial_guess = [np.array([2, 4]), 6]
        formatted_guess = MeasurementFunction._format_initial_guess(initial_guess)
        npt.assert_array_equal(formatted_guess[0], np.array([2,4]))
        npt.assert_array_equal(formatted_guess[1], np.array([6]))