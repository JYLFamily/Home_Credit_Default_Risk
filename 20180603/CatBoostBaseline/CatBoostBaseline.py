# coding:utf-8

import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


class CatBoostBaseline(object):

    def __init__(self, *, input_path, output_path, output_file_name):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__output_file_name = output_file_name

        self.__train, self.__test = [None for _ in range(2)]
        self.__sample_submission = None
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None
        self.__categorical_index = None

        self.__cat = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_feature_df.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path, "sample_submission_one.csv"))

        # self.__train = self.__train.drop("SK_ID_CURR", axis=1)
        # self.__test = self.__test.drop("SK_ID_CURR", axis=1)

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop("TARGET", axis=1)
        self.__test_feature = self.__test

        self.__categorical_index = np.where(self.__train_feature.dtypes == "object")[0]
        self.__test_feature = self.__test_feature[self.__train_feature.columns]
        self.__train_feature.iloc[:, self.__categorical_index] = self.__train_feature.iloc[:, self.__categorical_index].fillna("missing")
        self.__test_feature.iloc[:, self.__categorical_index] = self.__test_feature.iloc[:, self.__categorical_index].fillna("missing")

    def model_fit(self):
        self.__cat = CatBoostClassifier(iterations=300)
        self.__cat.fit(self.__train_feature, self.__train_label, cat_features=self.__categorical_index)

    def model_predict(self):
        self.__sample_submission["TARGET"] = self.__cat.predict_proba(self.__test_feature)[:, 1]
        self.__sample_submission.to_csv(os.path.join(self.__output_path, self.__output_file_name), index=False)


if __name__ == "__main__":
    cat = CatBoostBaseline(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
        output_file_name=sys.argv[3]
    )
    cat.data_prepare()
    cat.model_fit()
    cat.model_predict()