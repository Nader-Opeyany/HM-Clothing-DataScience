class ThreePartDataset:
    def __init__(self, articles, customers, transactions, prune=False):
        self.articles = articles
        self.customers = customers
        self.transactions = transactions
        if prune:
            self.prune()

    def prune(self):
        self.articles = prune_articles(self.articles, transactions=self.transactions)
        self.customers = prune_customers(self.customers, transactions=self.transactions)


def prune_customers(customers, customer_ids=None, transactions=None):
    if transactions is not None:
        customer_ids = transactions.customer_id.unique()
    return customers.loc[customers.customer_id.isin(customer_ids), :]


def prune_articles(articles, article_ids=None, transactions=None):
    if transactions is not None:
        article_ids = transactions.article_id.unique()
    return articles.loc[articles.article_id.isin(article_ids), :]
