import numpy as np
import pandas as pd

from hmcollab import three_part_dataset
from hmcollab.articles import ArticleFeatureMunger


def customer_id(i):
    return f"0{i:x}"


def fake_customers(n, r=None):
    if r is None:
        r = np.random.RandomState(42)
    ids = [customer_id(i) for i in range(n)]
    ages = r.randint(100, size=n)
    return pd.DataFrame.from_dict({"customer_id": ids, "age": ages})


def articles_random_df(n=20, r=None):
    if r is None:
        r = np.random.RandomState(42)
    article_id = ["0" + str(i) for i in range(n)]
    color = ["black", "white"]
    article_color = [color[r.randint(low=0, high=len(color))] for i in range(n)]
    article_type = ["shirt", "pants", "shoes"]
    article = [article_type[r.randint(low=0, high=len(article_type))] for i in range(n)]
    return pd.DataFrame(
        {"article_id": article_id, "color": article_color, "article": article}
    )


def dates_random(n_month=50, r=None):
    if r is None:
        r = np.random.RandomState(42)
    years = [2021, 2022]
    months = {"03": 31, "04": 30}
    dates = []
    for year in years:
        for month, month_max in months.items():
            days = r.randint(low=1, high=month_max, size=n_month)
            temp = [
                str(year) + "-" + str(month) + "-0" + str(day)
                if day < 10
                else str(year) + "-" + str(month) + "-" + str(day)
                for day in days
            ]
            dates.extend(temp)
    return dates


def transactions_random_df(customer_df, articles_df, n=100, r=None):
    if r is None:
        r = np.random.RandomState(42)
    article_ids = r.choice(articles_df.article_id.unique(), size=n)
    customer_ids = r.choice(customer_df.customer_id.unique(), size=n)
    dates = dates_random(n_month=n, r=r)
    r.shuffle(dates)
    dates = dates[:n]
    prices = r.rand(n)
    sales_channels = r.choice([1, 2], size=n)
    df = pd.DataFrame.from_dict(
        {
            "t_dat": dates,
            "customer_id": customer_ids,
            "article_id": article_ids,
            "price": prices,
            "sales_channel_id": sales_channels,
        }
    )
    df.t_dat = pd.to_datetime(df.t_dat)
    return df


def random_dataset(n_customers=100, n_articles=100, n_transactions=1000, r=None):
    if r is None:
        r = np.random.RandomState(42)
    customers = fake_customers(n_customers, r=r)
    articles = articles_random_df(n_articles, r=r)
    transactions = transactions_random_df(customers, articles, n_transactions, r=r)
    return three_part_dataset.ThreePartDataset(articles, customers, transactions)
