import unittest

import pandas as pd
from sklearn.cluster import KMeans

import hmcollab.splitter
from hmcollab import articles
from hmcollab import transactions

from hmcollab.tests import fake_data


class TestTransactions(unittest.TestCase):
    def setUp(self):
        self.customer = "05a"
        self.clusters = 2
        self.days = 7

        self.dataset = fake_data.random_dataset()

    def tearDown(self):
        pass

    def test_slit_by_time_first_six(self):
        six_transactions = self.dataset.transactions.iloc[:6].copy()
        x, y = hmcollab.splitter.split_by_time(six_transactions, self.days)

        # verify that returned dataframes have correct columns
        expected = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
        actual = list(y.columns)
        self.assertEqual(expected, actual)

        actual = list(x.columns)
        self.assertEqual(expected, actual)

        # verify timestamps
        actual = list(y.t_dat)
        expected = [
            pd.Timestamp(year=2022, month=3, day=25),
            pd.Timestamp(year=2022, month=3, day=23),
        ]
        self.assertEqual(expected, actual)

        actual = list(x.t_dat)
        expected = [
            pd.Timestamp(year=2021, month=4, day=25),
            pd.Timestamp(year=2021, month=4, day=3),
            pd.Timestamp(year=2021, month=4, day=25),
            pd.Timestamp(year=2021, month=3, day=14),
        ]
        self.assertEqual(expected, actual)

        # verify customer ids
        actual = list(y.customer_id)
        expected = ["010", "02a"]
        self.assertEqual(expected, actual)

        actual = list(x.customer_id)
        expected = ["05a", "021", "05a", "03a"]
        self.assertEqual(expected, actual)

        # verify article ids
        actual = list(y.article_id)
        expected = ["015", "089"]
        self.assertEqual(expected, actual)

        actual = list(x.article_id)
        expected = ["041", "098", "06", "059"]
        self.assertEqual(expected, actual)

        # verify price
        actual = list(y.price)
        expected = [0.09468099050479506, 0.032856697740396346]
        self.assertEqual(expected, actual)

        actual = list(x.price)
        expected = [
            0.28816452288395134,
            0.6039204436622274,
            0.049593508220086124,
            0.49916983326266173,
        ]
        self.assertEqual(expected, actual)

        # verify sales channel id
        actual = list(y.sales_channel_id)
        expected = [1, 2]
        self.assertEqual(expected, actual)

        actual = list(x.sales_channel_id)
        expected = [1, 1, 1, 1]
        self.assertEqual(expected, actual)

    def test_all_article_ids(self):
        t = transactions.TransactionsByCustomer(self.dataset.transactions)
        actual = t.all_article_ids(self.customer)
        actual = list(actual)
        expected = ["041", "06", "016", "028", "056", "073"]
        self.assertEqual(expected, actual)

    def test_kmeans_consumer_2(self):
        # Note: Do not scale dummy features
        munger = articles.ArticleFeatureMungerSpecificFeatures(
            self.dataset.articles,
            features=[
                "color",
                "article",
            ],
            use_article_id=True,
        )
        full_dummies = munger.x
        t = transactions.TransactionsByCustomer(self.dataset.transactions)
        basket = t.all_article_ids(self.customer)
        customer_dummies = full_dummies.merge(
            basket, on="article_id", how="right"
        ).drop(columns="article_id")
        kmeans = KMeans(
            init="k-means++",
            n_clusters=self.clusters,
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        expected = kmeans.fit(customer_dummies).cluster_centers_
        actual = transactions.kmeans_consumer(
            customer_dummies, k=self.clusters
        ).cluster_centers_

        self.assertEqual(expected.tolist(), actual.tolist())
