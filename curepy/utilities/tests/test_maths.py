"""Tests for utilities.maths"""

import unittest

from curepy.utilities.maths import lnlike


class TestMaths(unittest.TestCase):

    def test_lnlike_basic(self):
        self.assertEqual(lnlike(4.0), -2.0)


if __name__ == "__main__":
    unittest.main()
