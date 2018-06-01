# coding:utf-8

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IndicateFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns, postfix="indicator"):
        self.columns = columns
        self.postfix = postfix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.copy()
        for col in self.columns:
            x[str(col) + "_" + self.postfix] = x[col].isna().astype(np.float64)

        return x

    def fit_transform(self, X, y=None, **fit_params):
        x = X.copy()
        for col in self.columns:
            x[str(col) + "_" + self.postfix] = x[col].isna().astype(np.float64)

        return x