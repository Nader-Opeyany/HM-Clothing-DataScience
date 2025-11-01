import datetime
import unittest

import numpy as np

from hmcollab import splitter
from hmcollab.three_part_dataset import prune_articles, prune_customers
from hmcollab.tests import fake_data


class TestSplitter(unittest.TestCase):
    def setUp(self):
        self.fake_dataset = fake_data.random_dataset(
            n_customers=100, n_articles=1000, n_transactions=10000
        )
        self.fake_dataset.prune()

    def tearDown(self):
        pass

    def test_split_by_time(self):
        transactions = self.fake_dataset.transactions
        expected = transactions.iloc[0].t_dat
        older, newer = splitter.split_by_time(transactions, days=20)
        actual = transactions.iloc[0].t_dat
        self.assertEqual(expected, actual)

        self.assertEqual(transactions.shape[0], older.shape[0] + newer.shape[0])
        new_min = newer.t_dat.min()
        new_max = newer.t_dat.max()
        old_min = older.t_dat.min()
        old_max = older.t_dat.max()

        self.assertGreater(new_min, old_max)

        self.assertEqual(new_max, transactions.t_dat.max())
        self.assertEqual(old_min, transactions.t_dat.min())

        cutoff_date = new_max - datetime.timedelta(days=20)
        self.assertLess(old_max, cutoff_date)

    def twenty_day_older_newer_split_test(self, older, newer):
        self.assertEqual(
            self.fake_dataset.transactions.shape[0], older.shape[0] + newer.shape[0]
        )
        new_min = newer.t_dat.min()
        new_max = newer.t_dat.max()
        old_min = older.t_dat.min()
        old_max = older.t_dat.max()

        self.assertGreater(new_min, old_max)

        self.assertEqual(new_max, self.fake_dataset.transactions.t_dat.max())
        self.assertEqual(old_min, self.fake_dataset.transactions.t_dat.min())

        cutoff_date = new_max - datetime.timedelta(days=20)
        self.assertLess(old_max, cutoff_date)

    def test_older_newer_portions_transactions(self):
        op = splitter.OlderPortion(20).split(self.fake_dataset)
        np = splitter.NewerPortion(20).split(self.fake_dataset)
        older = op.transactions
        newer = np.transactions
        self.twenty_day_older_newer_split_test(older, newer)

    def test_prune_customers(self):
        customer_df = self.fake_dataset.customers
        customer_ids = {
            "02",
            "03",
            "05",
        }
        pruned_customers = prune_customers(customer_df, customer_ids=customer_ids)
        expected = customer_ids
        actual = set(pruned_customers.customer_id.unique())
        self.assertEqual(expected, actual)

    def test_prune_articles(self):
        article_ids = {
            "02",
            "03",
            "05",
            "07",
        }
        pruned_articles = prune_articles(
            self.fake_dataset.articles, article_ids=article_ids
        )
        expected = article_ids
        actual = set(pruned_articles.article_id.unique())
        self.assertEqual(expected, actual)

    def test_split_by_ids(self):
        customer_df = fake_data.fake_customers(20)
        full_ids = customer_df.customer_id.unique()
        full_ids = set(full_ids)
        ids_train, ids_test = splitter.split_ids(customer_df, 0.2)
        self.assertEqual(len(full_ids), len(ids_train) + len(ids_test))
        actual = set(ids_test)
        expected = {"00", "01", "011", "0f"}
        self.assertEqual(expected, actual)
        union = actual.union(ids_train)
        self.assertEqual(full_ids, union)

    def test_transactions_train_test(self):
        ids_train, ids_test = splitter.split_ids(self.fake_dataset.customers, 0.2)
        train, test = splitter.transactions_train_test(
            self.fake_dataset.transactions, ids_train, ids_test
        )
        self.assertEqual(
            self.fake_dataset.transactions.shape[0], train.shape[0] + test.shape[0]
        )

        expected = set(ids_test)
        actual = set(test.customer_id.unique())
        self.assertEqual(expected, actual)

        expected = set(ids_train)
        actual = set(train.customer_id.unique())
        self.assertEqual(expected, actual)

    def test_customer_portion(self):
        all_customer_ids = self.fake_dataset.customers.customer_id.unique()
        r = np.random.RandomState(42)
        customer_ids = r.choice(all_customer_ids, size=4, replace=False)
        cp = splitter.CustomerPortion(customer_ids)
        ds = cp.split(self.fake_dataset)

        actual = ds.customers.shape
        expected = (4, 2)
        self.assertEqual(expected, actual)

        actual = ds.transactions.shape
        expected = (383, 5)
        self.assertEqual(expected, actual)

        actual = ds.articles.shape
        expected = (333, 3)
        self.assertEqual(expected, actual)

        expected = set(customer_ids)
        actual = set(ds.customers.customer_id.unique())
        self.assertEqual(expected, actual)

        actual = set(ds.transactions.customer_id.unique())
        self.assertEqual(expected, actual)

        expected = set(ds.articles.article_id.unique())
        actual = set(ds.transactions.article_id.unique())
        self.assertEqual(expected, actual)

    def test_split_by_customer_train_test_strategy_default(self):
        strategy = splitter.SplitByCustomerTrainTestStrategy(self.fake_dataset, 0.2)
        full_ids = self.fake_dataset.customers.customer_id.unique()
        full_ids = set(full_ids)

        # test that test and train customers have expected shapes
        expected = self.fake_dataset.customers.shape[0]
        actual = strategy.train.customers.shape[0] + strategy.test.customers.shape[0]
        self.assertEqual(expected, actual)

        expected = self.fake_dataset.customers.shape[1]
        actual = strategy.train.customers.shape[1]
        self.assertEqual(expected, actual)

        actual = strategy.test.customers.shape[1]
        self.assertEqual(expected, actual)

        # test that test and train articles have expected shapes
        expected = self.fake_dataset.articles.shape[1]
        actual = strategy.train.articles.shape[1]
        self.assertEqual(expected, actual)

        actual = strategy.test.articles.shape[1]
        self.assertEqual(expected, actual)

        # test that test and train transactions have expected shapes
        expected = self.fake_dataset.transactions.shape[0]
        actual = (
            strategy.train.transactions.shape[0] + strategy.test.transactions.shape[0]
        )
        self.assertEqual(expected, actual)

        expected = self.fake_dataset.transactions.shape[1]
        actual = strategy.train.transactions.shape[1]
        self.assertEqual(expected, actual)

        actual = strategy.test.transactions.shape[1]
        self.assertEqual(expected, actual)

        ids_train = strategy.train.customers.customer_id.unique()
        ids_test = strategy.test.customers.customer_id.unique()
        actual = set(ids_test)
        expected = {
            "0a",
            "01f",
            "04",
            "0c",
            "053",
            "046",
            "01e",
            "021",
            "04d",
            "016",
            "012",
            "050",
            "02d",
            "049",
            "04c",
            "035",
            "05a",
            "00",
            "027",
            "02c",
        }
        self.assertEqual(expected, actual)
        union = actual.union(ids_train)
        self.assertEqual(full_ids, union)

        ids_train = set(strategy.train.transactions.customer_id.unique())
        ids_test = set(strategy.test.transactions.customer_id.unique())
        self.assertEqual(0, len(ids_train.intersection(ids_test)))
        actual = set(ids_test)
        self.assertEqual(expected, actual)
        union = actual.union(ids_train)
        self.assertEqual(full_ids, union)

        full_ids = self.fake_dataset.articles.article_id.unique()
        full_ids = set(full_ids)
        ids_train = strategy.train.articles.article_id.unique()
        ids_test = strategy.test.articles.article_id.unique()
        union = set(ids_test).union(ids_train)
        self.assertEqual(full_ids, union)

    def test_time_split_strategy(self):
        time_spliter = splitter.TimeSplitStrategy(20)
        op = time_spliter.older_portion.split(self.fake_dataset)
        np = time_spliter.newer_portion.split(self.fake_dataset)
        older = op.transactions
        newer = np.transactions
        self.twenty_day_older_newer_split_test(older, newer)

    def test_xy_strategy(self):
        xy_strategy = splitter.XYStrategy(self.fake_dataset, 20)
        self.twenty_day_older_newer_split_test(
            xy_strategy.x.transactions, xy_strategy.y.transactions
        )

    def test_standard_strategy(self):
        standard_strategy = splitter.StandardStrategy(self.fake_dataset, 20)
        self.assertEqual(
            self.fake_dataset.transactions.shape[0],
            standard_strategy.x.transactions.shape[0]
            + standard_strategy.vy.transactions.shape[0]
            + standard_strategy.y.transactions.shape[0],
        )
        older = standard_strategy.x.transactions
        middle = standard_strategy.vy.transactions
        newer = standard_strategy.y.transactions

        new_min = newer.t_dat.min()
        new_max = newer.t_dat.max()
        mid_min = middle.t_dat.min()
        mid_max = middle.t_dat.max()
        old_min = older.t_dat.min()
        old_max = older.t_dat.max()

        self.assertGreater(new_min, mid_max)
        self.assertGreater(mid_min, old_max)

        self.assertEqual(new_max, self.fake_dataset.transactions.t_dat.max())
        self.assertEqual(old_min, self.fake_dataset.transactions.t_dat.min())

        cutoff_date = new_max - datetime.timedelta(days=20)
        self.assertLess(mid_max, cutoff_date)

        cutoff_date = mid_max - datetime.timedelta(days=20)
        self.assertLess(old_max, cutoff_date)

    def test_customer_train_test_val_strategy(self):
        strategy = splitter.SplitByCustomerTrainTestValidationStrategy(
            self.fake_dataset, 0.2, 0.2
        )
        full_ids = self.fake_dataset.customers.customer_id.unique()
        full_ids = set(full_ids)
        train_ids = strategy.train.customers.customer_id.unique()
        train_ids = set(train_ids)
        test_ids = strategy.test.customers.customer_id.unique()
        test_ids = set(test_ids)
        val_ids = strategy.validation.customers.customer_id.unique()
        val_ids = set(val_ids)

        expected = full_ids
        actual = train_ids.union(test_ids).union(val_ids).union(train_ids)
        self.assertEqual(expected, actual)

        expected = set()
        actual = train_ids.intersection(test_ids)
        self.assertEqual(expected, actual)

        actual = train_ids.intersection(val_ids)
        self.assertEqual(expected, actual)

        actual = test_ids.intersection(val_ids)
        self.assertEqual(expected, actual)
