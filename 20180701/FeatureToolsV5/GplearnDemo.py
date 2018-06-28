# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
np.random.seed(7)


class GplearnDemo(object):
    def __init__(self):
        # data prepare
        self.__boston = None
        self.__boston_feature = None
        self.__boston_label = None
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label, self.__test_label = [None for _ in range(2)]
        self.__transformer = None
        self.__gp_train_feature = None
        self.__gp_test_feature = None

        # model fit
        self.__regressor = None

    def data_prepare(self):
        self.__boston = load_boston()
        self.__boston_feature = pd.DataFrame(self.__boston.data, columns=self.__boston.feature_names)
        self.__boston_label = pd.Series(self.__boston.target).to_frame("TARGET").squeeze()

        self.__train_feature, self.__test_feature, self.__train_label, self.__test_label = (
            train_test_split(
                self.__boston_feature,
                self.__boston_label,
                test_size=0.5,
                shuffle=True
            )
        )

        # 不能有缺失值
        self.__transformer = SymbolicTransformer(n_jobs=4)
        self.__transformer.fit(self.__train_feature, self.__train_label)
        self.__gp_train_feature = self.__transformer.transform(self.__train_feature)
        self.__gp_test_feature = self.__transformer.transform(self.__test_feature)

    def model_fit_predict(self):
        self.__regressor = Ridge()
        self.__regressor.fit(self.__train_feature, self.__train_label)
        print(mean_squared_error(self.__test_label, self.__regressor.predict(self.__test_feature)))

        self.__regressor = Ridge()
        self.__regressor.fit(np.hstack((self.__train_feature.values, self.__gp_train_feature)), self.__train_label)
        print(mean_squared_error(self.__test_label, self.__regressor.predict(np.hstack((self.__test_feature.values, self.__gp_test_feature)))))


if __name__ == "__main__":
    gd = GplearnDemo()
    gd.data_prepare()
    gd.model_fit_predict()