"""Tests for RetrievalInput class"""

import unittest
import numpy as np
import numpy.testing as npt
from unittest.mock import patch

from curepy.container.retrieval_input import RetrievalInput

class TestRetrievalInput(unittest.TestCase):
    def test_init_builds_empty(self):
        inp = RetrievalInput()
        self.assertIsNone(inp.prior_obj)
        self.assertIsNone(inp.measurement_obj)
        self.assertIsNone(inp.ancillary_obj)
        self.assertIsNone(inp.measurement_function_obj)