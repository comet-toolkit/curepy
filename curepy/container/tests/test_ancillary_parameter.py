"""Tests for Ancillary class"""

import unittest
import numpy as np
from unittest.mock import patch

from curepy.container.ancillary_parameter import AncillaryParameter

class TestAncillary(unittest.TestCase):
    def test_init_builds_empty_ancill(self):
        a = AncillaryParameter()
        self.assertIsNone(a.b)