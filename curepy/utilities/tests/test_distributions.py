"""Tests for utilities.distributions"""

import unittest
import numpy as np

from curepy.utilities.distributions import (
    ln_uniform,
    ln_normal,
    ln_multi_normal,
)


class TestDistributions(unittest.TestCase):

    def test_ln_uniform_scalar_outside(self):
        self.assertEqual(ln_uniform(1.5, 0.0, 1.0), -np.inf)

    def test_ln_uniform_scalar_inside(self):
        self.assertEqual(ln_uniform(0.5, 0.0, 1.0), 0)

    def test_ln_uniform_array(self):
        theta = np.array([0.2, 0.8])
        self.assertEqual(ln_uniform(theta, 0.0, 1.0), 0)

    def test_ln_normal(self):
        val = ln_normal(1.0, 0.0, 2.0)
        expected = -0.5 * ((1.0 - 0.0) ** 2) / (2 * 2.0**2)
        self.assertEqual(val, expected)

    def test_ln_multi_normal_1d(self):
        theta = np.array([1.0])
        mu = np.array([0.0])
        Sa_inv = np.array([[2.0]])
        val = ln_multi_normal(theta, mu, Sa_inv)
        expected = -0.5 * (1.0**2) * 2.0
        self.assertEqual(val, expected)

    def test_ln_multi_normal_2d(self):
        theta = np.array([1.0, 2.0])
        mu = np.array([0.0, 0.0])
        Sa_inv = np.eye(2)
        val = ln_multi_normal(theta, mu, Sa_inv)
        expected = -0.5 * (theta.T @ Sa_inv @ theta)
        self.assertEqual(val, expected)


if __name__ == "__main__":
    unittest.main()
