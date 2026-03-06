"""Tests for MeasurementFunction class"""

import unittest
import numpy as np
import numpy.testing as npt
from unittest.mock import patch

from curepy.container.measurement_function import MeasurementFunction

def dummy_function(x, b):
    return x + b

def dummy_function_2(x, b1, b2):
    return x + b1 - b2
    
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

    def test_make_x_tuple_list(self):
        initial_guess = [0.0, 0.0, 0.0]
        mf = MeasurementFunction(dummy_function, initial_guess)
        out = mf.make_x_tuple([10, 20, 30]) 
        self.assertEqual(out, (10, 20, 30))

    def test_make_x_tuple_1d_array(self):
        initial_guess = [0.0, 0.0, 0.0]
        mf = MeasurementFunction(dummy_function, initial_guess)
        out = mf.make_x_tuple(np.array([10, 20, 30])) 
        self.assertEqual(out, (10, 20, 30))

    def test_make_x_tuple_ragged_list(self):
        initial_guess = [[0, 0], [0]]
        mf = MeasurementFunction(dummy_function, initial_guess)
        out = mf.make_x_tuple(np.array([10, 20, 30])) 
        npt.assert_array_equal(out[0], np.array([10,20]))
        self.assertEqual(out[1], np.array([30]))

    def test__make_x_tuple_ragged_numpy_array(self):
        initial_guess = np.array([np.array([0, 0]), np.array([0])], dtype=object)
        mf = MeasurementFunction(dummy_function, initial_guess)
        out = mf.make_x_tuple([9, 8, 7])
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], np.ndarray) and isinstance(out[1], np.ndarray)
        assert np.array_equal(out[0], np.array([9, 8]))
        assert np.array_equal(out[1], np.array([7]))

    def test_make_x_tuple_too_deep_nesting_raises(self):
        initial_guess = [[[[0]]]]
        mf = MeasurementFunction(dummy_function, initial_guess)
        with self.assertRaises(ValueError):
            mf.make_x_tuple([1])

    def test_make_x_tuple_theta_too_short_raises(self):
        initial_guess = [0, 0, 0]
        mf = MeasurementFunction(dummy_function, initial_guess)
        with self.assertRaises(IndexError):
            mf.make_x_tuple([1, 2])
            
    def test_measurement_function_x(self):
        theta = np.array([2,3,4])
        b = [1]
        mf = MeasurementFunction(dummy_function, [[0,0,0]])
        out = mf.measurement_function_x(theta, b)
        npt.assert_array_equal(out, np.array([3,4,5]))
        
    def test_measurement_function_flattened_output(self):
        theta = np.array([2,3,4,5])
        b = [1]
        mf = MeasurementFunction(dummy_function, [[[0,0], [0,0]]])
        out = mf.measurement_function_flattened_output(theta, b)
        npt.assert_array_equal(out, np.array([3,4,5,6]))
        
    def test_measurement_function_flattened_b(self):
        theta = [6]
        b_flat = np.array([1,2,3])
        mf = MeasurementFunction(dummy_function_2, [0])
        out = mf.measurement_function_flattened_b(theta, b_flat, [(2,),(1,)])
        npt.assert_array_equal(out, np.array([4,5]))