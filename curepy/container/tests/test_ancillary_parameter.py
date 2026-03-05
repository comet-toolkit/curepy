"""Tests for Ancillary class"""

import unittest
import numpy as np
from unittest.mock import patch
import numpy.testing as npt

from curepy.container.ancillary_parameter import AncillaryParameter

class TestAncillary(unittest.TestCase):
    def test_init_builds_empty_ancill(self):
        a = AncillaryParameter()
        self.assertIsNone(a.b)
        
    def test_calculate_b_cov_none(self):
        a = AncillaryParameter()
        with self.assertWarns(Warning):
            cov = a.calculate_b_cov()
        self.assertIsNone(cov)

    def test_calculate_b_cov_no_corr(self):
        b = np.ones(5)
        u_b = b * 0.1
        a = AncillaryParameter(b, u_b)
        with self.assertWarns(Warning):
            cov = a.calculate_b_cov()
        self.assertIsNone(cov)
        
    def test_calculate_b_cov_single_b(self):
        b = np.ones(5)
        u_b = b * 0.1
        corr_b = [np.eye(len(b))]

        a = AncillaryParameter(b, u_b, corr_b)
        cov = a.calculate_b_cov()
        npt.assert_array_almost_equal(cov, 0.01*np.eye(len(b)))
        
    def test_calculate_b_cov_equal_len_b(self):
        b = [np.ones(5), 2*np.ones(5)]
        u_b = [0.1*np.ones(5), 0.2*np.ones(5)]
        corr_b = [np.ones((5, 5)), np.eye(5)]

        a = AncillaryParameter(b, u_b, corr_b, corr_between_b=np.eye(2))
        cov = a.calculate_b_cov()
        npt.assert_array_almost_equal(cov, np.block([[0.01*np.ones((5,5)), np.zeros((5,5))],[np.zeros((5,5)),0.04*np.eye(5)]]))
        
    def test_calculate_b_cov_unequal_len_b(self):
        b = [np.ones(5), 2*np.ones(2)]
        u_b = [0.1*np.ones(5), 0.2*np.ones(2)]
        corr_b = [np.ones((5, 5)), np.eye(2)]

        a = AncillaryParameter(b, u_b, corr_b)
        cov = a.calculate_b_cov()
        npt.assert_array_almost_equal(cov, np.block([[0.01*np.ones((5,5)), np.zeros((5,2))],[np.zeros((2,5)),0.04*np.eye(2)]]))