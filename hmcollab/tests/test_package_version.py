import unittest

import hmcollab


class TestPackage(unittest.TestCase):
    def test_something(self):
        self.assertIsNotNone(hmcollab.__version__)
