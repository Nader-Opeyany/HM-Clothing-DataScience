import unittest

from hmcollab import directories
from hmcollab.directory_tree import HMDatasetDirectoryTree


class TestDirectoryTree(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_directory_tree(self):
        tree = HMDatasetDirectoryTree(base="base")
        expected = "base"
        actual = tree.path()
        self.assertEqual(expected, actual)

        expected = "base/offset"
        actual = tree.path("offset")
        self.assertEqual(expected, actual)

        expected = "base/articles.csv"
        actual = tree.articles
        self.assertEqual(expected, actual)

        expected = "base/customers.csv"
        actual = tree.customers
        self.assertEqual(expected, actual)

        expected = "base/transactions_train.csv"
        actual = tree.transactions
        self.assertEqual(expected, actual)

        expected = "base/images/123/1234567890.jpg"
        actual = tree.image(1234567890)
        self.assertEqual(expected, actual)

        expected = directories.data()
        actual = HMDatasetDirectoryTree().path()
        self.assertEqual(expected, actual)
