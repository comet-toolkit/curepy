"""Tests for Prior class"""

import unittest
import numpy as np
from unittest.mock import patch

from curepy.container.prior import Prior

shape = ["normal", "normal"]
params = [{"mu": 5, "sigma": 3}, {"mu": 0, "sigma": 1}]
corr = np.eye(len(shape))


class TestPrior(unittest.TestCase):
    def test_init_builds_prior(self):
        p = Prior(["uniform"], [{"minimum": -np.inf, "maximum": np.inf}])

        self.assertEqual(p.prior_params, [{"minimum": -np.inf, "maximum": np.inf}])

    def test__check_inputs_pass(self):
        shape = ["normal", "normal"]
        params = [{"mu": 5, "sigma": 3}, {"mu": 0, "sigma": 1}]
        corr = np.eye(len(shape))
        Prior._check_inputs(shape, params, corr)

    def test__check_inputs_empty(self):
        with self.assertRaises(ValueError):
            Prior._check_inputs(None, [{}], [])

    def test__check_inputs_invalid_shape(self):
        with self.assertRaises(ValueError):
            Prior._check_inputs(
                ["normal", "invalid"], [{"mu": 0, "sigma": 1}, {"mu": 0}], None
            )

    def test__check_inputs_invalid_params(self):
        with self.assertRaises(ValueError):
            Prior._check_inputs(["normal"], [{"mu": 0, "invalid": 1}], None)

    def test__check_inputs_invalid_correlation(self):
        with self.assertRaises(ValueError):
            Prior._check_inputs(
                ["normal", "uniform"],
                [{"mu": 0, "sigma": 1}, {"maximum": 0, "minimum": 1}],
                np.eye(2),
            )

    def test__check_inputs_no_correlation_warning(self):
        with self.assertWarns(Warning):
            Prior._check_inputs(
                ["normal", "normal"],
                [{"mu": 0, "sigma": 1}, {"mu": 0, "sigma": 1}],
                None,
            )

    @patch("curepy.utilities.utilities.format_correlation")
    def test_return_Sa_inv_none(self, mock_format):
        mock_format.return_value = None
        p = Prior(
            ["uniform"],
            [
                {"minimum": 0, "maximum": 1},
            ],
        )
        Sa_inv = p.return_Sa_inv()
        self.assertIsNone(Sa_inv)

    @patch("curepy.utilities.utilities.format_correlation")
    def test_return_Sa_inv(self, mock_format):
        mock_format.return_value = np.eye(2)
        p = Prior(
            ["normal"],
            [
                {"mu": 0, "sigma": 1},
            ],
        )
        Sa_inv = p.return_Sa_inv()
        np.testing.assert_array_equal(Sa_inv, np.eye(2))
