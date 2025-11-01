import numpy as np

from hmcollab.similarity import Similarity, IdenticalSimilarity


def relevant(predicted, target, similarity: Similarity = None):
    """
    :param predicted: DataFrame (columns: customer_id and prediction)
        Recommendations for each customer. The prediction column contains
        space separated recommendations
    :param target: DataFrame (columns: customer_id and last_7d)
        All customers with at least one transaction in the last 7 days. The column
        last_7d contains space separated article_id from all transactions from a particular
        customer
    :param similarity: A similarity measurement. Expected to have a function compare_one.
        Will be set to IdentitySimilarity if None.
    :return: Boolean series
        True if the article_id is present in the target dataset for that particular customer
    """

    if similarity is None:
        similarity = IdenticalSimilarity()

    def compare_one(p, r):
        return similarity.compare_one(p.split(" "), r.split(" "))

    # Keep only customers with transactions at target set (inner)
    both = predicted.merge(target, on=["customer_id"], how="inner")
    return both.apply(lambda x: compare_one(x.prediction, x.target), axis=1)


def precision_at_k(ranked_results):
    return sum(ranked_results) / len(ranked_results)


def ap_at_k(ranked_results):
    count = len(ranked_results)
    if not count:
        return 0
    s = 0
    for i in range(len(ranked_results)):
        if ranked_results[i]:
            s += precision_at_k(ranked_results[: i + 1])
    return s / count


def map_at_k(ranked_results):
    return np.mean([ap_at_k(r) for r in ranked_results])
