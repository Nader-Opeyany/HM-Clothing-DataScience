import numpy as np
import pandas as pd

import hmcollab.splitter
from .directory_tree import HMDatasetDirectoryTree
from .three_part_dataset import ThreePartDataset


def target_to_relevant(trans_y):
    """Convert to dataframe of customers with a list of transactions from the input set"""

    def relevant_dict(tup):
        return " ".join(tup[1].loc[:, "article_id"].tolist())

    grouped = trans_y.loc[:, ["customer_id", "article_id"]].groupby(["customer_id"])
    by_row = {t[0]: relevant_dict(t) for t in grouped}
    relevant_set = pd.DataFrame.from_dict(by_row, orient="index", columns=["target"])
    relevant_set.reset_index(inplace=True)
    relevant_set.rename(columns={"index": "customer_id"}, inplace=True)

    return relevant_set


class Target:
    def __init__(self, transactions_df, days=7):
        self.transactions = transactions_df
        self.transactions_x, self.transactions_y = hmcollab.splitter.split_by_time(
            self.transactions, days=days
        )
        self.relevant_set = target_to_relevant(self.transactions_y)


class HMDataset(ThreePartDataset):
    def __init__(
        self,
        threepartdataset=None,
        tree=None,
        articles=None,
        customers=None,
        transactions=None,
        relevant_set=None,
        folds="twosets",
        prune=False,
    ):
        if threepartdataset is not None:
            articles = threepartdataset.articles
            customers = threepartdataset.customers
            transactions = threepartdataset.transactions
            relevant_set = None

        if articles is None:
            if tree is None:
                tree = HMDatasetDirectoryTree()
            self.tree = tree
            articles, customers, transactions, relevant_set = tree.load()

        ThreePartDataset.__init__(self, articles, customers, transactions, prune=prune)

        target = Target(self.transactions)
        self.transactions_x, self.transactions_y = (
            target.transactions_x,
            target.transactions_y,
        )

        if relevant_set is None:
            relevant_set = target.relevant_set
        self.relevant_set = relevant_set

        if folds == "twosets":
            ids_train, ids_test = hmcollab.splitter.split_ids(
                self.transactions, fraction=0.2
            )
            self.train_x, self.test_x = hmcollab.splitter.transactions_train_test(
                self.transactions_x, ids_train, ids_test
            )
            self.train_y, self.test_y = hmcollab.splitter.transactions_train_test(
                self.transactions_y, ids_train, ids_test
            )

        if folds == "threesets":
            # Train: 60%, val: 20%, test: 20%
            # Creating test set
            random_state = np.random.RandomState(42)
            ids_train, ids_test = hmcollab.splitter.split_ids(
                self.transactions, fraction=0.2, random_state=random_state
            )
            train_x, self.test_x = hmcollab.splitter.transactions_train_test(
                self.transactions_x, ids_train, ids_test
            )
            train_y, self.test_y = hmcollab.splitter.transactions_train_test(
                self.transactions_y, ids_train, ids_test
            )
            # Creating training and validation sets
            ids_train, ids_val = hmcollab.splitter.split_ids(
                train_x, fraction=0.25, random_state=random_state
            )  # 25% of training is 20% from the total: 80(.25)=20
            self.train_x, self.val_x = hmcollab.splitter.transactions_train_test(
                train_x, ids_train, ids_val
            )
            self.train_y, self.val_y = hmcollab.splitter.transactions_train_test(
                train_y, ids_train, ids_val
            )

        if folds == "standard":
            # Splitting by leave last week. Note that train_x is used to train all (train, validations and test)
            # train_vy is the target variable for validation to use with train data
            self.train_y = self.transactions_y
            self.train_x, self.train_vy = hmcollab.splitter.split_by_time(
                self.transactions_x, days=7
            )


class HMDatasetTwoSets(HMDataset):
    def __init__(
        self,
        threepartdataset=None,
        tree=None,
        articles=None,
        customers=None,
        transactions=None,
        relevant_set=None,
        prune=False,
    ):
        HMDataset.__init__(
            self,
            threepartdataset=threepartdataset,
            tree=tree,
            articles=articles,
            customers=customers,
            transactions=transactions,
            relevant_set=relevant_set,
            folds="twosets",
            prune=prune,
        )


class HMDatasetThreeSets(HMDataset):
    def __init__(
        self,
        threepartdataset=None,
        tree=None,
        articles=None,
        customers=None,
        transactions=None,
        relevant_set=None,
        prune=False,
    ):
        HMDataset.__init__(
            self,
            threepartdataset=threepartdataset,
            tree=tree,
            articles=articles,
            customers=customers,
            transactions=transactions,
            relevant_set=relevant_set,
            folds="threesets",
            prune=prune,
        )


class HMDatasetStandard(HMDataset):
    def __init__(
        self,
        threepartdataset=None,
        tree=None,
        articles=None,
        customers=None,
        transactions=None,
        relevant_set=None,
        prune=False,
    ):
        HMDataset.__init__(
            self,
            threepartdataset=threepartdataset,
            tree=tree,
            articles=articles,
            customers=customers,
            transactions=transactions,
            relevant_set=relevant_set,
            folds="standard",
            prune=prune,
        )
