# coding:utf-8

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
np.random.seed(7)


class CatBoostBaseline(object):

    def __init__(self, *, path):
        self.__path = path
        self.__application_train = None
        self.__application_test = None
        self.__sample_submission = None

        self.__application_train_feature = None
        self.__application_train_label = None
        self.__application_test_feature = None

        self.__categorical_columns = None
        self.__numeric_columns = None

        self.__categorical_index = None

        self.__clf = None

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__path, "application_train.csv"))
        self.__application_test = pd.read_csv(os.path.join(self.__path, "application_test.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__path, "sample_submission_one.csv"))

        self.__application_train = self.__application_train.drop("SK_ID_CURR", axis=1)
        self.__application_test = self.__application_test.drop("SK_ID_CURR", axis=1)

        self.__application_train_feature = self.__application_train[[i for i in self.__application_train.columns if i != "TARGET"]]
        self.__application_train_label = self.__application_train["TARGET"]
        self.__application_test_feature = self.__application_test

        self.__categorical_columns = self.__application_train_feature.select_dtypes(include=["object"]).columns.tolist()
        self.__numeric_columns = [i for i in self.__application_train_feature.columns if i not in self.__categorical_columns]

        self.__application_train_feature[self.__categorical_columns] = (
            self.__application_train_feature[self.__categorical_columns].fillna("missing")
        )
        self.__application_train_feature[self.__numeric_columns] = (
            self.__application_train_feature[self.__numeric_columns].fillna(-999)
        )
        self.__application_test_feature[self.__categorical_columns] = (
            self.__application_test_feature[self.__categorical_columns].fillna("missing")
        )
        self.__application_test_feature[self.__numeric_columns] = (
            self.__application_test_feature[self.__numeric_columns].fillna(-999)
        )

        self.__categorical_index = np.where(self.__application_train_feature.dtypes == "object")[0]

    def model_fit(self):
        self.__clf = CatBoostClassifier(iterations=300)
        self.__clf.fit(self.__application_train_feature, self.__application_train_label, cat_features=self.__categorical_index)

    def model_predict(self):
        self.__sample_submission["TARGET"] = self.__clf.predict_proba(self.__application_test_feature)[:, 1]
        self.__sample_submission.to_csv(
            "C:\\Users\\puhui\\PycharmProjects\\Home_Credit_Default_Risk\\20180520\CatBoostBaseline\\sample_submission.csv",
            index=False
        )


if __name__ == "__main__":
    cbb = CatBoostBaseline(path="C:\\Users\\puhui\\PycharmProjects\\Home_Credit_Default_Risk\\Data")
    cbb.data_prepare()
    cbb.model_fit()
    cbb.model_predict()