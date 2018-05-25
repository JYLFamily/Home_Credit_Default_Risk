# coding:utf-8

import numpy as np


class CatBoostDataPrepare(object):

    def __init__(self, *, train_feature, test_feature):
        self.__train_feature = train_feature.copy()
        self.__test_feature = test_feature.copy()
        self.__categorical_index = None
        self.__numeric_index = None

    def data_prepare(self):
        """ 离散变量 缺失值填充 missing 连续变量 缺失值填充 +999/-999
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

        # 让 catboost 自行处理缺失值

        # self.__train_feature.iloc[:, self.__numeric_index] = (
        #     self.__train_feature.iloc[:, self.__numeric_index].apply(
        #         lambda x: x.fillna(-999.0) if x.median() > 0 else x.fillna(999.0)
        #     )
        # )
        # self.__test_feature.iloc[:, self.__numeric_index] = (
        #     self.__test_feature.iloc[:, self.__numeric_index].apply(
        #         lambda x: x.fillna(-999.0) if x.median() > 0 else x.fillna(999.0)
        #     )
        # )

        return self.__train_feature, self.__test_feature