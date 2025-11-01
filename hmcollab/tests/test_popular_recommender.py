import unittest

from hmcollab import datasets, articles, models
from hmcollab.tests import fake_data


class TestPopularRecommender(unittest.TestCase):
    def setUp(self):
        dataset = fake_data.random_dataset(
            n_customers=3, n_articles=10, n_transactions=100
        )
        self.dataset = datasets.HMDatasetTwoSets(threepartdataset=dataset)
        self.dataset.prune()
        self.articles_munger = self.get_simple()
        self.customer = self.dataset.customers.customer_id[1]
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

    def test_popular_recommender(self):
        recommender = models.PopularRecommender(
            self.dataset,
            total_recommendations=2,
        )

        # 1. Test recommend
        actual_recommend = recommender.recommend()
        expected = ["07", "09"]

        self.assertEqual(expected, actual_recommend)

        actual_recommend_all = recommender.recommend_all(self.customer_list)

        actual_rows = actual_recommend_all.shape[0]
        expected = len(self.customer_list)
        self.assertEqual(expected, actual_rows)

        actual = actual_recommend_all.prediction.value_counts().shape[0]
        expected = 1
        self.assertEqual(expected, actual)

        actual = actual_recommend_all.iloc[0, 1]
        expected = " ".join(actual_recommend)
        self.assertEqual(expected, actual)
