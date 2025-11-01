import os

import pandas as pd

from hmcollab import directories


def read_with_article_id(filename):
    return pd.read_csv(
        filename,
        dtype={
            "article_id": object,
        },
    )


class HMDatasetDirectoryTree:
    def __init__(self, base=None):
        if base is None:
            base = directories.data()
        self._base = base

    def path(self, filename=None):
        return directories.qualifyname(self._base, filename)

    def images(self, filename=None):
        return directories.qualifyname(self.path("images"), filename)

    @property
    def customers(self):
        return self.path("customers.csv")

    @property
    def articles(self):
        return self.path("articles.csv")

    @property
    def transactions(self):
        return self.path("transactions_train.csv")

    @property
    def transactions_y_by_customer(self):
        return self.path("target_set_7d_75481u.csv")

    @property
    def transactions_y_by_customer_exists(self):
        return os.path.exists(self.transactions_y_by_customer)

    def image(self, number):
        number = str(number)
        filename = "{}.jpg".format(number)
        prefix = number[:3]
        dir = self.images(prefix)
        return os.path.join(dir, filename)

    def load_articles(self):
        return pd.read_csv(
            self.articles,
            dtype={
                "article_id": object,
                "product_code": object,
                "colour_group_code": object,
            },
        )

    def load_customers(self):
        return pd.read_csv(self.customers)

    def load_transactions(self):
        return read_with_article_id(self.transactions)

    def load_relevant(self):
        return read_with_article_id(self.transactions_y_by_customer)

    def load(self):
        articles = self.load_articles()
        customers = self.load_customers()

        relevant_set = None
        transactions = self.load_transactions()
        if self.transactions_y_by_customer_exists:
            relevant_set = self.load_relevant()

        return articles, customers, transactions, relevant_set
