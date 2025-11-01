import unittest

import pandas as pd

from hmcollab import datasets
from hmcollab.tests import fake_data


class TestHMDataset(unittest.TestCase):
    def setUp(self):
        self.fake_dataset = fake_data.random_dataset(
            n_customers=100, n_articles=1000, n_transactions=10000
        )
        self.fake_dataset.prune()

    def tearDown(self):
        pass

    def test_target(self):
        target = datasets.Target(self.fake_dataset.transactions, 20)

        expected = (8172, 5)
        actual = target.transactions_x.shape
        self.assertEqual(expected, actual)

        expected = (1828, 5)
        actual = target.transactions_y.shape
        self.assertEqual(expected, actual)

        expected = (100, 2)
        actual = target.relevant_set.shape
        self.assertEqual(expected, actual)

        expected = ["customer_id", "target"]
        actual = list(target.relevant_set.columns)
        self.assertEqual(expected, actual)

        self.assertEqual("03e", target.relevant_set.iloc[50].customer_id)
        expected = (
            "0882 0775 0817 0697 0848 0994 0684 0520 0982 0868 0834 0437 072 0659 0419"
        )
        actual = target.relevant_set.iloc[50].target
        self.assertEqual(expected, actual)

    def test_target_to_relevant(self):
        target = datasets.Target(self.fake_dataset.transactions, 20)
        relevant = datasets.target_to_relevant(target.transactions_y)

        expected = (100, 2)
        actual = relevant.shape
        self.assertEqual(expected, actual)

        expected = ["customer_id", "target"]
        actual = list(relevant.columns)
        self.assertEqual(expected, actual)

        self.assertEqual("03e", relevant.iloc[50].customer_id)

        expected = "0563 0225 0836 0658 0836 0402 0796 0310 0725 0250 0510 0216 0711 0319 0741 0133"
        actual = relevant.iloc[0].target
        self.assertEqual(expected, actual)

        expected = (
            "0882 0775 0817 0697 0848 0994 0684 0520 0982 0868 0834 0437 072 0659 0419"
        )
        actual = relevant.iloc[50].target
        self.assertEqual(expected, actual)

        self.assertEqual(
            list(target.relevant_set.customer_id), list(relevant.customer_id)
        )
        self.assertEqual(list(target.relevant_set.target), list(relevant.target))
        # TODO: Really? It seems that target.relevant_set is the same as relevant.

    def test_hmdataset_twosets(self):
        dataset = datasets.HMDatasetTwoSets(threepartdataset=self.fake_dataset)

        expected = (100, 2)
        actual = dataset.relevant_set.shape
        self.assertEqual(expected, actual)

        expected = list(self.fake_dataset.transactions.columns)
        for df in [dataset.train_x, dataset.train_y, dataset.test_x, dataset.test_y]:
            actual = list(df.columns)
            self.assertEqual(expected, actual)

        expected = (7391, 5)
        actual = dataset.train_x.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2021-03-18 00:00:00"),
            "customer_id": "07",
            "article_id": "0737",
            "price": 0.31984350550585794,
            "sales_channel_id": 1,
        }
        actual = dataset.train_x.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (564, 5)
        actual = dataset.train_y.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-04-23 00:00:00"),
            "customer_id": "05f",
            "article_id": "0348",
            "price": 0.1403088393250692,
            "sales_channel_id": 1,
        }
        actual = dataset.train_y.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (1903, 5)
        actual = dataset.test_x.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-03-26 00:00:00"),
            "customer_id": "02d",
            "article_id": "0429",
            "price": 0.8648273676096885,
            "sales_channel_id": 1,
        }
        actual = dataset.test_x.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (142, 5)
        actual = dataset.test_y.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-04-24 00:00:00"),
            "customer_id": "03d",
            "article_id": "074",
            "price": 0.40993047713908326,
            "sales_channel_id": 2,
        }
        actual = dataset.test_y.iloc[0].to_dict()
        self.assertEqual(expected, actual)

    def test_hmdataset_threesets(self):
        dataset = datasets.HMDatasetThreeSets(threepartdataset=self.fake_dataset)

        expected = list(self.fake_dataset.transactions.columns)
        for df in [
            dataset.train_x,
            dataset.train_y,
            dataset.val_x,
            dataset.val_y,
            dataset.test_x,
            dataset.test_y,
        ]:
            actual = list(df.columns)
            self.assertEqual(expected, actual)

        expected = (5546, 5)
        actual = dataset.train_x.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2021-03-18 00:00:00"),
            "customer_id": "07",
            "article_id": "0737",
            "price": 0.31984350550585794,
            "sales_channel_id": 1,
        }
        actual = dataset.train_x.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (419, 5)
        actual = dataset.train_y.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-04-23 00:00:00"),
            "customer_id": "046",
            "article_id": "0272",
            "price": 0.9131621714028072,
            "sales_channel_id": 2,
        }
        actual = dataset.train_y.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (1845, 5)
        actual = dataset.val_x.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2021-03-18 00:00:00"),
            "customer_id": "07",
            "article_id": "0737",
            "price": 0.31984350550585794,
            "sales_channel_id": 1,
        }
        actual = dataset.train_x.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (145, 5)
        actual = dataset.val_y.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-04-23 00:00:00"),
            "customer_id": "046",
            "article_id": "0272",
            "price": 0.9131621714028072,
            "sales_channel_id": 2,
        }
        actual = dataset.train_y.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (1903, 5)
        actual = dataset.test_x.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-03-26 00:00:00"),
            "customer_id": "02d",
            "article_id": "0429",
            "price": 0.8648273676096885,
            "sales_channel_id": 1,
        }
        actual = dataset.test_x.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (142, 5)
        actual = dataset.test_y.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-04-24 00:00:00"),
            "customer_id": "03d",
            "article_id": "074",
            "price": 0.40993047713908326,
            "sales_channel_id": 2,
        }
        actual = dataset.test_y.iloc[0].to_dict()
        self.assertEqual(expected, actual)

    def test_hmdataset_standard(self):
        dataset = datasets.HMDatasetStandard(threepartdataset=self.fake_dataset)

        expected = list(self.fake_dataset.transactions.columns)
        for df in [dataset.train_x, dataset.train_vy, dataset.train_y]:
            actual = list(df.columns)
            self.assertEqual(expected, actual)

        expected = (8624, 5)
        actual = dataset.train_x.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-03-26 00:00:00"),
            "customer_id": "02d",
            "article_id": "0429",
            "price": 0.8648273676096885,
            "sales_channel_id": 1,
        }
        actual = dataset.train_x.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (670, 5)
        actual = dataset.train_vy.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-04-17 00:00:00"),
            "customer_id": "04f",
            "article_id": "0975",
            "price": 0.6704246583783142,
            "sales_channel_id": 1,
        }
        actual = dataset.train_vy.iloc[0].to_dict()
        self.assertEqual(expected, actual)

        expected = (706, 5)
        actual = dataset.train_y.shape
        self.assertEqual(expected, actual)

        expected = {
            "t_dat": pd.Timestamp("2022-04-24 00:00:00"),
            "customer_id": "03d",
            "article_id": "074",
            "price": 0.40993047713908326,
            "sales_channel_id": 2,
        }

        actual = dataset.train_y.iloc[0].to_dict()
        self.assertEqual(expected, actual)
