# coding:utf-8

import numpy as np
from sklearn.preprocessing import LabelEncoder


class LgboostDataPrepare(object):

    def __init__(self, *, train_feature, test_feature):
        self.__train_feature = train_feature.copy()
        self.__test_feature = test_feature.copy()
        self.__categorical_index = None
        self.__numeric_index = None
        self.__encoder = None

    def data_prepare(self):
        """ 离散变量 缺失值填充 missing 后 labelencoder
        :return: 训练集特征 测试集特征
        """
        self.__categorical_index = np.where(self.__train_feature.dtypes == "object")[0]
        self.__numeric_index = np.where(self.__train_feature.dtypes != "object")[0]

        self.__train_feature.iloc[:, self.__categorical_index] = (
            self.__train_feature.iloc[:, self.__categorical_index].fillna("missing")
        )
        self.__test_feature.iloc[:, self.__categorical_index] = (
            self.__test_feature.iloc[:, self.__categorical_index].fillna("missing")
        )
        for i in self.__categorical_index:
            self.__encoder = LabelEncoder()
            self.__encoder.fit(self.__train_feature.iloc[:, i])
            self.__train_feature.iloc[:, i] = self.__encoder.transform(self.__train_feature.iloc[:, i])
            self.__test_feature.iloc[:, i] = (  # test 中存在 train 中没有的 categories
                ["missing" if i not in self.__encoder.classes_ else i for i in self.__test_feature.iloc[:, i]]
            )
            self.__test_feature.iloc[:, i] = self.__encoder.transform(self.__test_feature.iloc[:, i])

        return self.__train_feature, self.__test_feature