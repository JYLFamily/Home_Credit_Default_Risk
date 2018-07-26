# coding:utf-8

import os
import re
import sys
import tqdm
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder


def filter_nan_feature(feature):
    """
    :param feature: feature pd.Series
    :return:
    """
    return (np.sum(feature.isna()) / len(feature)) > 0.9


class FeatureSelection(object):
    def __init__(self, *, input_path, output_path):
        # init
        self.__input_path, self.__output_path = input_path, output_path

        # data prepare
        self.__train_feature_before, self.__train_feature_after = [None for _ in range(2)];
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_label = None
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__categorical_columns = None

        # data output
        self.__train_select_feature, self.__test_select_feature = [None for _ in range(2)]

    def data_prepare(self):
        self.__train_feature_before = pd.read_csv(os.path.join(self.__input_path, "train_feature_before_df.csv"))
        self.__train_feature_after = pd.read_csv(os.path.join(self.__input_path, "train_feature_after_df.csv"))
        self.__train = pd.concat([self.__train_feature_before, self.__train_feature_after])
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_feature_df.csv"))

        self.__train_label = self.__train["TARGET"].copy()
        self.__train_feature = (
            self.__train.drop(
                ["TARGET"] + [col for col in self.__train.columns.tolist() if re.search(r"SK_ID", col)], axis=1
            )
        ).copy()
        self.__test_feature = self.__test[self.__train_feature.columns.tolist()].copy()
        self.__categorical_columns = self.__train_feature.select_dtypes(include="object").columns.tolist()

        encoder = TargetEncoder()
        encoder.fit(self.__train_feature[self.__categorical_columns], self.__train_label)
        self.__train_feature[self.__categorical_columns] = encoder.transform(
            self.__train_feature[self.__categorical_columns]
        )

    def feature_filter(self):
        # np.nan feature filter
        flag_list = []
        for col in tqdm.tqdm(self.__train_feature.columns):
            flag_list.append(filter_nan_feature(self.__train_feature[col]))
        self.__train_feature = self.__train_feature[
            [col for col, flag in zip(self.__train_feature.columns, flag_list) if flag is not True]]

        # std filter
        flag_list = []
        for col in tqdm.tqdm(self.__train_feature.columns):
            flag_list.append(self.__train_feature[col].std() < 0.01)
        self.__train_feature = self.__train_feature[
            [col for col, flag in zip(self.__train_feature.columns, flag_list) if flag is not True]]

    def data_output(self):
        self.__train_select_feature = (
            self.__train[["TARGET"] + self.__train_feature.columns.tolist()]
        )
        self.__test_select_feature = (
            self.__test[self.__train_feature.columns.tolist()]
        )

        self.__train_select_feature.to_csv(
            os.path.join(self.__output_path, "train_select_feature_df.csv"), index=False
        )
        self.__test_select_feature.to_csv(
            os.path.join(self.__output_path, "test_select_feature_df.csv"), index=False
        )

if __name__ == "__main__":
    fs = FeatureSelection(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    fs.data_prepare()
    fs.feature_filter()
    fs.data_output()