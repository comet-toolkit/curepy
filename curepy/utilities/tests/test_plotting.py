"""Tests for utilities.plotting"""

import unittest
import numpy as np

from curepy.utilities.plotting import plot_corner, quantile, hist2d


class TestPlotting(unittest.TestCase):

    def test_quantile_basic(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        q = quantile(x, [0.5])
        self.assertEqual(float(q), 1.5)

    def test_quantile_invalid_q(self):
        x = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            quantile(x, [-0.1])

    def test_quantile_weights_mismatch(self):
        x = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            quantile(x, [0.5], weights=np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
