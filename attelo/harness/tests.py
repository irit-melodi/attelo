"""
attelo.harness tests
"""

import unittest

from .example import TinyHarness


# pylint: disable=too-few-public-methods

# pylint: disable=no-self-use
class HarnessTest(unittest.TestCase):
    """
    Tests for the attelo.harness infrastructure
    """
    def test_run_harness(self):
        """Check that the harness does not crash on example data
        """
        TinyHarness().run()
