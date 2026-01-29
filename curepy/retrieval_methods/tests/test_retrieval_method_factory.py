"""Tests for retrieval method factory"""

import unittest

from curepy.retrieval_methods.retrieval_method_factory import RetrievalFactory

class TestRetrievalFactory(unittest.TestCase):
    def test_get_brdf_model_valid_name(self):
        factory = RetrievalFactory()
        model = factory.make_retrieval_object("mcmc", nwalkers = 100, burn_in = 100, steps = 1000)
        self.assertEqual(model.__class__.__name__, "MCMC")

    def test_get_brdf_model_invalid_name(self):
        factory = RetrievalFactory()
        with self.assertRaises(ValueError):
            model = factory.make_retrieval_object("invalid", nwalkers = 100, burn_in = 100, steps = 1000)