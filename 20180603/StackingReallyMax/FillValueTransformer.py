# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class FillValueTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, filling_values=0):
        self.filling_values=filling_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.copy()
        return x.fillna(self.filling_values)

    def fit_transform(self, X, y=None, **fit_params):
        x = X.copy()
        return x.fillna(self.filling_values)


if __name__ == "__main__":
    # pl = Pipeline(steps=[
    #     ("Indicator", IndicateFeatureTransformer()),
    #     ("FillValue", FillValueTransformer(filling_values=-999))
    # ])
    # X = pd.DataFrame([[np.nan, 4, np.nan], [1, 2, 3]])
    # pl.fit(X)
    # print(pl.transform(X))

    pass


