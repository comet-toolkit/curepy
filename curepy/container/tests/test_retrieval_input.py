"""Tests for RetrievalInput class"""

import unittest
import numpy as np

from curepy.container.retrieval_input import RetrievalInput

class TestRetrievalInput(unittest.TestCase):
    def test_init_builds_empty(self):
        inp = RetrievalInput()
        self.assertIsNone(inp.prior_obj)
        self.assertIsNone(inp.measurement_obj)
        self.assertIsNone(inp.ancillary_obj)
        self.assertIsNone(inp.measurement_function_obj)
        
    def test_build_prior(self):
        inp = RetrievalInput()
        inp.build_prior(['uniform'], [{'minimum':0,'maximum':1}])
        self.assertIsNotNone(inp.prior_obj)
        
    def test_build_ancillary(self):
        inp = RetrievalInput()
        inp.build_ancillary()
        self.assertIsNotNone(inp.ancillary_obj)
    
    def test_build_measurement(self):
        inp = RetrievalInput()
        inp.build_measurement(np.array([4]))
        self.assertIsNotNone(inp.measurement_obj)
        
    def test_build_measurement_function(self):
        inp = RetrievalInput()
        inp.build_measurement_function(lambda a : np.array([a]), [9])
        self.assertIsNotNone(inp.measurement_function_obj)
        
    def test_init_build_retrieval_inputs(self):
        inp = RetrievalInput()
        inp.build_retrieval_inputs(measurement_func=lambda a : np.array([a]),
                                   initial_guess=[8],
                                   y = np.array([4]),
                                   prior_shape=['uniform'],
                                   prior_params=[{'minimum':0,'maximum':10}])
        self.assertIsNotNone(inp.prior_obj)
        self.assertIsNotNone(inp.measurement_obj)
        self.assertIsNotNone(inp.ancillary_obj)
        self.assertIsNotNone(inp.measurement_function_obj)    