# coding:utf-8

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy


class TrainTestFeatureDistribution(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path = input_path
        self.__output_path = output_path

        # data prepare
        self.__train = None
        self.__test = None

        self.__train_feature = None
        self.__test_feature = None

        self.__numeric_columns = None
        self.__categorical_columns = None

        # calc entropy
        self.__numeric_entropy_df = None
        self.__categorical_entropy_df = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_feature_df.csv"))

        self.__train_feature = self.__train.drop(["SK_ID_CURR", "TARGET"], axis=1)
        self.__test_feature = self.__test
        self.__test_feature = self.__test_feature[self.__train_feature.columns]

        self.__numeric_columns = self.__train_feature.select_dtypes(exclude="object").columns
        self.__categorical_columns = self.__train_feature.select_dtypes(include="object").columns

    def calc_entropy(self):

        self.__numeric_entropy_df = pd.DataFrame(columns=["feature", "non_entropy", "nan_entropy"])
        for col in self.__numeric_columns:
            # 非缺失分位数
            p_non = np.nanpercentile(a=self.__train_feature[col], q=list(range(1, 101, 1)))
            q_non = np.nanpercentile(a=self.__test_feature[col], q=list(range(1, 101, 1)))
            # 缺失比例, 非缺失比例
            p_nan = [
                np.sum(self.__train_feature[col].isna()) / len(self.__train_feature[col]),
                1 - np.sum(self.__train_feature[col].isna()) / len(self.__train_feature[col])
            ]
            q_nan = [
                np.sum(self.__test_feature[col].isna()) / len(self.__test_feature[col]),
                1 - np.sum(self.__test_feature[col].isna()) / len(self.__test_feature[col])
            ]
            # 拼接行
            row = pd.DataFrame(
                [[col, entropy(p_non, q_non), entropy(p_nan, q_nan)]],
                columns=["feature", "non_entropy", "nan_entropy"]
            )
            self.__numeric_entropy_df = self.__numeric_entropy_df.append(row)

    def show_entropy(self):
        self.__numeric_entropy_df = self.__numeric_entropy_df.sort_values(by="non_entropy", ascending=False)
        print(self.__numeric_entropy_df.tail(10))


if __name__ == "__main__":
    ttfd = TrainTestFeatureDistribution(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\feature_data_V2",
        output_path=None
    )
    ttfd.data_prepare()
    ttfd.calc_entropy()
    ttfd.show_entropy()