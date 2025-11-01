from abc import ABCMeta, abstractmethod

import numpy as np


class Similarity(metaclass=ABCMeta):
    @abstractmethod
    def similarity(self, id0, id1):
        pass

    def compare_one(self, predicted, target) -> np.array:
        def is_in_target(i):
            for j in target:
                if self.similarity(i, j):
                    return True
            return False

        comparisons = np.array([is_in_target(i) for i in predicted])
        return comparisons


class IdenticalSimilarity(Similarity):
    def similarity(self, id0, id1):
        return id0 == id1

    def compare_one(self, predicted, target):
        return np.isin(predicted, target)


class ArticleSimilarityByColumn(Similarity):
    def __init__(self, df, column_name):
        self.df = df
        self.column_name = column_name

    def row_from_article_id(self, id0):
        df = self.df[self.df.article_id == id0]
        if df.shape[0]:
            return df.iloc[0]

    def similarity(self, id0, id1):
        row0 = self.row_from_article_id(id0)
        if row0 is None:
            return False
        row1 = self.row_from_article_id(id1)
        if row1 is None:
            return False
        return self.similarity_by_row(row0, row1)

    def similarity_by_iloc(self, index0, index1):
        row0 = self.df.iloc[index0]
        row1 = self.df.iloc[index1]
        return self.similarity_by_row(row0, row1)

    def similarity_by_row(self, row0, row1):
        return row0[self.column_name] == row1[self.column_name]


def get_similarity(similarity_name, articles_df):
    if similarity_name in [
        "product_code",
        "colour_group_code",
        "department_no",
        "garment_group_no",
    ]:
        return ArticleSimilarityByColumn(articles_df, similarity_name)

    return IdenticalSimilarity()
