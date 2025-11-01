import datetime
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hmcollab.three_part_dataset import (
    ThreePartDataset,
    prune_customers,
    prune_articles,
)


def split_by_time(df, days):
    """Split transactions dataframe by a cutoff number of days from
    the last transaction at column t_dat"""
    df["t_dat"] = pd.to_datetime(df["t_dat"].copy(), format="%Y-%m-%d")
    last = df["t_dat"].max()
    cutoff_date = last - datetime.timedelta(days=days)
    older = df[df.t_dat < cutoff_date]
    newer = df[df.t_dat >= cutoff_date]
    return older, newer


def split_ids(a_set, fraction, random_state=42):
    full_ids = a_set.customer_id.unique()
    ids_train, ids_test = train_test_split(
        full_ids, test_size=fraction, random_state=random_state
    )
    return ids_train, ids_test


def transactions_train_test(a_set, ids_tr, ids_te):
    # a_set: transactions dataset
    train = a_set.loc[a_set.customer_id.isin(ids_tr), :]
    test = a_set.loc[a_set.customer_id.isin(ids_te), :]
    return train, test


class Portion(metaclass=ABCMeta):
    def split(self, dataset):
        transactions = self._get_transaction_portion(dataset)
        customers = prune_customers(dataset.customers, transactions=transactions)
        articles = prune_articles(dataset.articles, transactions=transactions)
        return ThreePartDataset(articles, customers, transactions)

    @abstractmethod
    def _get_transaction_portion(self, dataset):
        pass


class OlderPortion(Portion):
    def __init__(self, days):
        self.days = days

    def _get_transaction_portion(self, dataset):
        older, newer = split_by_time(dataset.transactions, self.days)
        return older


class NewerPortion(Portion):
    def __init__(self, days):
        self.days = days

    def _get_transaction_portion(self, dataset):
        older, newer = split_by_time(dataset.transactions, self.days)
        return newer


class TimeSplitStrategy:
    def __init__(self, days):
        self.newer_portion = NewerPortion(days)
        self.older_portion = OlderPortion(days)


class XYStrategy(TimeSplitStrategy):
    def __init__(self, dataset: ThreePartDataset, days):
        super().__init__(days)

        older = self.older_portion.split(dataset)
        newer = self.newer_portion.split(dataset)
        self.partition = [older, newer]

    @property
    def x(self):
        return self.partition[0]

    @property
    def y(self):
        return self.partition[1]


class StandardStrategy(TimeSplitStrategy):
    def __init__(self, dataset: ThreePartDataset, days):
        super().__init__(days)

        older = self.older_portion.split(dataset)
        newer = self.newer_portion.split(dataset)
        oldest = self.older_portion.split(older)
        middle = self.newer_portion.split(older)
        self.partition = [oldest, middle, newer]

    @property
    def x(self):
        return self.partition[0]

    @property
    def vy(self):
        return self.partition[1]

    @property
    def y(self):
        return self.partition[2]


class CustomerPortion(Portion):
    def __init__(self, customer_ids):
        self.customer_ids = customer_ids

    def _get_transaction_portion(self, dataset):
        return dataset.transactions.loc[
            dataset.transactions.customer_id.isin(self.customer_ids), :
        ]

    def split(self, dataset: ThreePartDataset):
        transactions = self._get_transaction_portion(dataset)
        customers = prune_customers(dataset.customers, customer_ids=self.customer_ids)
        articles = prune_articles(dataset.articles, transactions=transactions)
        return ThreePartDataset(articles, customers, transactions)


class SplitByCustomerStrategy:
    def __init__(self, dataset: ThreePartDataset, customer_id_partition):
        partition = []
        for ids in customer_id_partition:
            cp = CustomerPortion(ids)
            partition.append(cp.split(dataset))
        self.partition = partition

    @staticmethod
    def _get_random_state(random_state):
        if type(random_state) is int:
            random_state = np.random.RandomState(random_state)
        return random_state


class SplitByCustomerTrainTestStrategy(SplitByCustomerStrategy):
    def __init__(self, dataset: ThreePartDataset, fraction, random_state=42):
        random_state = SplitByCustomerStrategy._get_random_state(random_state)
        ids_train, ids_test = split_ids(
            dataset.customers, fraction, random_state=random_state
        )
        super().__init__(dataset, [ids_train, ids_test])

    @property
    def train(self):
        return self.partition[0]

    @property
    def test(self):
        return self.partition[1]


class SplitByCustomerTrainTestValidationStrategy(SplitByCustomerStrategy):
    def __init__(
        self,
        dataset: ThreePartDataset,
        test_fraction,
        validation_fraction,
        random_state=42,
    ):
        random_state = SplitByCustomerStrategy._get_random_state(random_state)
        full_ids = dataset.customers.customer_id.unique()
        partial_ids, ids_test = train_test_split(
            full_ids, test_size=test_fraction, random_state=random_state
        )
        ids_train, ids_validation = train_test_split(
            partial_ids,
            test_size=validation_fraction / (1.0 - test_fraction),
            random_state=random_state,
        )
        super().__init__(dataset, [ids_train, ids_test, ids_validation])

    @property
    def train(self):
        return self.partition[0]

    @property
    def test(self):
        return self.partition[1]

    @property
    def validation(self):
        return self.partition[2]
