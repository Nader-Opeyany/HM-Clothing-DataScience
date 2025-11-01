from abc import ABCMeta, abstractmethod

import pandas as pd


class ArticleMunger(metaclass=ABCMeta):
    def __init__(self, df: pd.DataFrame, use_article_id=False):
        self.df = df
        self.use_article_id = use_article_id
        self.x = self._to_array()

    @abstractmethod
    def _to_array(self):
        pass

    def id_from_index(self, i):
        return self.df.iloc[i].article_id


class ArticleFeatureMunger(ArticleMunger, metaclass=ABCMeta):
    @abstractmethod
    def features(self):
        pass

    def _to_array(self):
        features = self.features()
        if self.use_article_id:
            return pd.get_dummies(
                self.df[["article_id"] + features], columns=features, prefix=features
            )
        return pd.get_dummies(self.df[features], columns=features, prefix=features)


class ArticleFeatureMungerSpecificFeatures(ArticleFeatureMunger):
    def __init__(self, df: pd.DataFrame, features: list, use_article_id=False):
        self.__features = features
        super().__init__(df, use_article_id)

    def features(self):
        return self.__features


class ArticleFeaturesSimpleFeatures(ArticleFeatureMunger):
    def features(self):
        return [
            "product_type_no",
            "product_group_name",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
            "department_no",
            "index_code",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
