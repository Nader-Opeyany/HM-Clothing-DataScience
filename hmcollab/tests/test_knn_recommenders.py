import unittest

import numpy as np

from hmcollab import datasets, articles, models
from hmcollab.tests import fake_data


class TestKNNRecommenders(unittest.TestCase):
    def setUp(self):
        dataset = fake_data.random_dataset(
            n_customers=3, n_articles=10, n_transactions=100
        )
        self.dataset = datasets.HMDatasetTwoSets(threepartdataset=dataset)
        self.dataset.prune()
        self.articles_munger = self.get_simple()
        self.customer = self.dataset.customers.customer_id[1]
        self.clusters = 2
        self.days = 7
        self.full_dummies = self.get_simple().x

        self.customer_list = [
            self.customer,
            self.dataset.customers.customer_id[2],
        ]

    def tearDown(self):
        pass

    def get_simple(self):
        return articles.ArticleFeatureMungerSpecificFeatures(
            self.dataset.articles,
            [
                "color",
                "article",
            ],
            use_article_id=True,
        )

    def get_threefold_dataset(self):
        dataset = fake_data.random_dataset(
            n_customers=3, n_articles=10, n_transactions=1000
        )

        dataset = datasets.HMDatasetThreeSets(threepartdataset=dataset)
        return dataset

    def test_filter_article(self):
        actual = models.filter_articles(
            self.dataset.train_x, threshold=4, warning=False
        )
        expected = ["00", "01", "02", "03", "06", "07", "08", "09"]
        self.assertEqual(expected, actual)

        # actual = models.filter_articles(self.dataset.train_x, threshold=5, warning=False)
        # expected = ["02", "03", "06", "07", "08", "09"]
        # self.assertEqual(expected, actual)

        # actual = models.filter_articles(self.dataset.train_x, threshold=6, warning=False)
        # expected = ["02", "06", "07", "08", "09"]
        # self.assertEqual(expected, actual)

        # actual = models.filter_articles(self.dataset.train_x, threshold=7, warning=False)
        # expected = ["07", "09"]
        # self.assertEqual(expected, actual)

        # self.assertEqual(
        #     9, self.dataset.train_x[self.dataset.train_x.article_id == "07"].shape[0]
        # )
        # self.assertEqual(
        #     8, self.dataset.train_x[self.dataset.train_x.article_id == "09"].shape[0]
        # )

    def test_knn_recommender_init(self):
        recommender = models.KnnRecommender(
            self.dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=6,
            threshold=6,
            warning=False,
        )
        self.assertEqual(2, recommender.groups)
        self.assertEqual(6, recommender.total_recommendations)
        self.assertEqual(5, recommender.filtered_dummies.article_id.nunique())
        self.assertEqual(3, recommender.recomendations_by_group)
        self.assertEqual(3, recommender.model.k)

        expected = [0, 1, 2]
        actual = recommender.model.nearest(np.array([0, 0, 0, 0, 0]))[1]
        actual = list(actual.reshape(-1))
        self.assertEqual(expected, actual)

    def test_knn_recommender_for3_init(self):
        dataset = self.get_threefold_dataset()
        recommender = models.KnnRecommender_for3(
            dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=6,
            threshold=6,
            warning=False,
        )
        self.assertEqual(2, recommender.groups)
        self.assertEqual(6, recommender.total_recommendations)
        self.assertEqual(10, recommender.filtered_dummies.article_id.nunique())
        self.assertEqual(3, recommender.recomendations_by_group)
        self.assertEqual(3, recommender.model.k)

        expected = [2, 0, 1]
        actual = recommender.model.nearest(np.array([0, 0, 0, 0, 0]))[1]
        actual = list(actual.reshape(-1))
        self.assertEqual(expected, actual)

    def test_knn_recommender_for3_init_val(self):
        dataset = self.get_threefold_dataset()
        recommender = models.KnnRecommender_for3(
            dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=6,
            threshold=0,
            split="val",
            warning=False,
        )
        self.assertEqual(2, recommender.groups)
        self.assertEqual(6, recommender.total_recommendations)
        self.assertEqual(10, recommender.filtered_dummies.article_id.nunique())
        self.assertEqual(3, recommender.recomendations_by_group)
        self.assertEqual(3, recommender.model.k)

        expected = [2, 0, 1]
        actual = recommender.model.nearest(np.array([0, 0, 0, 0, 0]))[1]
        actual = list(actual.reshape(-1))
        self.assertEqual(expected, actual)

    def test_knn_recommender_for3_init_test(self):
        dataset = self.get_threefold_dataset()
        recommender = models.KnnRecommender_for3(
            dataset,
            self.full_dummies,
            groups=2,
            total_recommendations=6,
            threshold=0,
            split="test",
            warning=False,
        )
        self.assertEqual(2, recommender.groups)
        self.assertEqual(6, recommender.total_recommendations)
        self.assertEqual(10, recommender.filtered_dummies.article_id.nunique())
        self.assertEqual(3, recommender.recomendations_by_group)
        self.assertEqual(3, recommender.model.k)

        expected = [2, 0, 1]
        actual = recommender.model.nearest(np.array([0, 0, 0, 0, 0]))[1]
        actual = list(actual.reshape(-1))
        self.assertEqual(expected, actual)
