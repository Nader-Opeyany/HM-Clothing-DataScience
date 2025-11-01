import unittest

import numpy as np
from numpy.linalg import norm

import hmcollab.models
from hmcollab import directories
from hmcollab import articles
from hmcollab.tests.fake_data import articles_random_df


# The suite test the following:
# + articles dataset and preprocessing such as:
#   shape and one-hot encoding implementation
#   Those tests can be replaced with unittest using
#   One-hot and indices can be tested with a simple synthetic dataset
#   consisting of a couple of categorical columns and a numerical index
#   column with some IDS starting with 0
# + Other are integration test testing results from KNN
#    We might want to remove those and only keep integration test are models


class TestArticles(unittest.TestCase):
    def setUp(self):
        self.simple_onehot = np.load(directories.testdata("simple_onehot.npy"))
        self.articles = articles_random_df(17)

    def tearDown(self):
        pass

    def get_simple(self):
        return articles.ArticleFeatureMungerSpecificFeatures(
            self.articles,
            [
                "color",
                "article",
            ],
        )

    def get_simple_knn(self):
        return hmcollab.models.ArticleKNN(self.get_simple().x, 4)

    def test_article_simple_feature_array(self):
        a = self.get_simple()
        expected = (17, 5)
        actual = a.x.shape
        self.assertEqual(expected, actual)

        # number of rows in original dataframe should be same as in matrix representation
        expected, _ = a.df.shape
        actual, _ = a.x.shape
        self.assertEqual(expected, actual)

        # test that actual onehot values are the same as a previously saved example
        expected = self.simple_onehot
        actual = a.x.values
        self.assertEqual(0, norm(actual - expected))

    def test_id_from_index(self):
        a = self.get_simple()
        expected = "02"
        actual = a.id_from_index(2)
        self.assertEqual(expected, actual)

    def knn_test(self, d, indices):
        # check that actual distances match expected
        expected = np.array([0.0, 0.0, 0.0, 0])
        actual = np.array(d[0])
        self.assertAlmostEqual(0, norm(actual - expected))

        # check that actual indices match expected
        expected = {3, 4, 6, 10}
        actual = set(indices[0])
        self.assertEqual(expected, actual)

    def test_knn_by_row(self):
        knn = self.get_simple_knn()
        x = self.simple_onehot

        # choose row 10, for which there are two other exact matches
        row = x[10]
        d, indices = knn.nearest(row=row)

        self.knn_test(d, indices)

    def test_knn_by_index(self):
        a = self.get_simple()
        knn = self.get_simple_knn()
        x = self.simple_onehot

        # choose row 10, for which there are two other exact matches
        row = a.x.values[10]

        d, indices = knn.nearest(row)
        self.knn_test(d, indices)
