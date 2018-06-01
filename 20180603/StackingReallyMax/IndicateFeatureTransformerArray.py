# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IndicateFeatureTransformerArray(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.copy()
        for col in self.columns:
            x = np.hstack((
                x,
                pd.Series(x[:, col]).isna().astype(int).values.reshape((-1, 1))
            ))

        return x

    def fit_transform(self, X, y=None, **fit_params):
        x = X.copy()
        for col in self.columns:
            x = np.hstack((
                x,
                pd.Series(x[:, col]).isna().astype(int).values.reshape((-1, 1))
            ))

        return x