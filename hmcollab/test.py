#!/usr/bin/env python
import argparse
import unittest

from hmcollab.tests.test_articles import TestArticles
from hmcollab.tests.test_directory_tree import TestDirectoryTree
from hmcollab.tests.test_directories import TestDirectories
from hmcollab.tests.test_example import TestExample
from hmcollab.tests.test_fake_data import TestFakeData
from hmcollab.tests.test_hmdataset import TestHMDataset
from hmcollab.tests.test_knn_recommenders import TestKNNRecommenders
from hmcollab.tests.test_package_version import TestPackage
from hmcollab.tests.test_popular_recommender import TestPopularRecommender
from hmcollab.tests.test_scoring import TestScoring
from hmcollab.tests.test_similarity import TestSimilarity
from hmcollab.tests.test_splitter import TestSplitter
from hmcollab.tests.test_transactions import TestTransactions
from hmcollab.tests.test_relevant import TestRelevant

# integration tests
from hmcollab.tests.integration_tests.integration_test_data_exists import (
    IntegrationTestDataExists,
)
from hmcollab.tests.integration_tests.integration_test_models import IntegrationTestModels


class CountSuite(object):
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d: %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))


def suite(integration):
    s = CountSuite()

    s.add(TestArticles)
    s.add(TestDirectoryTree)
    s.add(TestDirectories)
    s.add(TestExample)
    s.add(TestFakeData)
    s.add(TestHMDataset)
    s.add(TestKNNRecommenders)
    s.add(TestScoring)
    s.add(TestPackage)
    s.add(TestPopularRecommender)
    s.add(TestSimilarity)
    s.add(TestSplitter)
    s.add(TestTransactions)
    s.add(TestRelevant)

    if integration:
        print("Running integration tests.")
        s.add(IntegrationTestDataExists)
        s.add(IntegrationTestModels)

    return s.s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests for HM Collab Project")
    parser.add_argument(
        "--integration",
        help="Whether to call tests which require dataset to be installed and generated files.",
        default=None,
    )
    args = parser.parse_args()
    runner = unittest.TextTestRunner()
    runner.run(suite(args.integration))
