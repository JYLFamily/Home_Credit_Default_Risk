# coding:utf-8

import numpy as np
import category_encoders as ce


class XgBoostDataPrepare(object):

    def __init__(self, *, train_feature, train_label, test_feature):
        self.__train_feature = train_feature.copy()
        self.__train_label = train_label.copy()
        self.__test_feature = test_feature.copy()
        self.__categorical_index = None
        self.__numeric_index = None
        self.__encoder = None

    def data_prepare(self):
        """ 离散变量 缺失值填充 missing 后均值编码连续化
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
        self.__encoder = ce.TargetEncoder()
        self.__encoder.fit(
            self.__train_feature.iloc[:, self.__categorical_index],
            self.__train_label
        )
        self.__train_feature.iloc[:, self.__categorical_index] = self.__encoder.transform(
            self.__train_feature.iloc[:, self.__categorical_index]
        )
        self.__test_feature.iloc[:, self.__categorical_index] = self.__encoder.transform(
            self.__test_feature.iloc[:, self.__categorical_index]
        )

        return self.__train_feature, self.__test_feature