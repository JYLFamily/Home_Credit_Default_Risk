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
        self.__feature_importance = None

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
        self.__feature_importance = pd.read_csv(
            "D:\\Code\\Python\\Home_Credit_Default_Risk\\20180617\\FeatureSelection\\train_feature_df_fs_mi.csv"
        )

        self.__train_feature = self.__train.drop(["SK_ID_CURR", "TARGET"], axis=1)
        self.__test_feature = self.__test

        self.__feature_importance = self.__feature_importance.loc[self.__feature_importance["mi"] > 0.01, "feature"]
        self.__train_feature = self.__train_feature[self.__feature_importance]
        self.__test_feature = self.__test_feature[self.__feature_importance]

        self.__numeric_columns = self.__train_feature.select_dtypes(exclude="object").columns
        self.__categorical_columns = self.__train_feature.select_dtypes(include="object").columns

    def calc_entropy(self):
        self.__numeric_entropy_df = pd.DataFrame(columns=["feature", "non_entropy", "nan_entropy"])
        for col in self.__numeric_columns:
            if len(list(set(list(self.__train_feature[col].dropna().unique()) + list(self.__test_feature[col].dropna().unique())))) <= 10:
                p_non = []
                q_non = []
                for level in list(set(list(self.__train_feature[col].dropna().unique()) + list(self.__test_feature[col].dropna().unique()))):
                    p_non.append(np.sum(self.__train_feature[col] == level) / len(self.__train_feature[col].dropna()))
                    q_non.append(np.sum(self.__test_feature[col] == level) / len(self.__test_feature[col].dropna()))
            else:
                # 非缺失分位数
                q = [0, 25, 50, 75, 100]
                p_percentile = np.nanpercentile(a=self.__train_feature[col], q=q)
                p_percentile = sorted(list(set(list(p_percentile))))
                p_non = (list(
                    pd.cut(self.__train_feature[col], bins=p_percentile)
                    .to_frame(col)
                    .groupby(col)[col].count() / np.sum(np.logical_not(self.__train_feature[col].isnull()))
                ))
                q_non = (list(
                    pd.cut(self.__test_feature[col], bins=p_percentile)
                    .to_frame(col)
                    .groupby(col)[col].count() / np.sum(np.logical_not(self.__test_feature[col].isnull()))
                ))

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

        self.__categorical_entropy_df = pd.DataFrame(columns=["feature", "non_entropy", "nan_entropy"])
        for col in self.__categorical_columns:
            p_non = []
            q_non = []
            for level in list(set(list(self.__train_feature[col].fillna("missing").unique()) + list(self.__test_feature[col].fillna("missing").unique()))):
                p_non.append(np.sum(self.__train_feature[col] == level) / len(self.__train_feature[col]))
                q_non.append(np.sum(self.__test_feature[col] == level) / len(self.__test_feature[col]))

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
            self.__categorical_entropy_df = self.__categorical_entropy_df.append(row)

    def show_entropy(self):
        self.__numeric_entropy_df = self.__numeric_entropy_df.sort_values(by="non_entropy", ascending=False)
        self.__numeric_entropy_df.to_csv(os.path.join(self.__output_path, "numeric_entropy_df.csv"), index=False)

        self.__categorical_entropy_df = self.__categorical_entropy_df.sort_values(by="non_entropy", ascending=False)
        self.__categorical_entropy_df.to_csv(os.path.join(self.__output_path, "categorical_entropy_df.csv"), index=False)


if __name__ == "__main__":
    ttfd = TrainTestFeatureDistribution(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\feature_data_V2",
        output_path="D:\\Kaggle\\Home_Credit_Default_Risk\\feature_data_V2"
    )
    ttfd.data_prepare()
    ttfd.calc_entropy()
    ttfd.show_entropy()