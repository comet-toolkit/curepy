"""Tests for utilities.utilities"""

import unittest
import numpy as np

from curepy.utilities.utilities import flatten_array, reshape_array, format_correlation


class TestUtilities(unittest.TestCase):

    def test_flatten(self):
        A = np.arange(6).reshape((2, 3))
        A2, shape = flatten_array(A)
        np.testing.assert_array_equal(A.flatten(), A2)

    def test_reshape(self):
        A = np.arange(6).reshape((2, 3))
        flat = np.arange(6)
        shape = (2,3)
        A2 = reshape_array(flat, shape)
        np.testing.assert_array_equal(A, A2)

    def test_format_correlation_none(self):
        y = np.arange(4)
        self.assertIsNone(format_correlation(None, None))

    def test_format_correlation_rand(self):
        y = np.arange(4)
        corr_rand = format_correlation(y, "rand")
        np.testing.assert_array_equal(corr_rand, np.eye(len(y)))

    def test_format_correlation_syst(self):
        y = np.arange(4)
        corr_syst = format_correlation(y, "syst")
        np.testing.assert_array_equal(corr_syst, np.ones((len(y), len(y))))

    def test_format_correlation_invalid_string(self):
        y = np.arange(3)
        with self.assertRaises(ValueError):
            format_correlation(y, "invalid")


if __name__ == "__main__":
    unittest.main()
