# coding:utf-8

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
np.random.seed(7)


class XgBoostBaseline(object):

    def __init__(self, *, path):
        self.__path = path
        self.__application_train = None
        self.__application_test = None
        self.__sample_submission = None

        self.__application_train_feature = None
        self.__application_train_label = None
        self.__application_test_feature = None

        self.__application_train_dmatrix = None
        self.__application_test_dmatrix = None

        self.__params = {}
        self.__clf = None

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__path, "application_train.csv"))
        self.__application_test = pd.read_csv(os.path.join(self.__path, "application_test.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__path, "sample_submission.csv"))

        self.__application_train = self.__application_train.drop("SK_ID_CURR", axis=1)
        self.__application_test = self.__application_test.drop("SK_ID_CURR", axis=1)

        self.__application_train_feature = self.__application_train[[i for i in self.__application_train.columns if i != "TARGET"]]
        self.__application_train_label = self.__application_train["TARGET"]
        self.__application_test_feature = self.__application_test

        self.__application_train_feature = self.__application_train_feature.drop(
            self.__application_train_feature.select_dtypes(include=["object"]).columns.tolist(),
            axis=1
        )
        self.__application_test_feature = self.__application_test_feature.drop(
            self.__application_test_feature.select_dtypes(include=["object"]).columns.tolist(),
            axis=1
        )

        self.__application_train_dmatrix = xgb.DMatrix(
            data=self.__application_train_feature,
            label=self.__application_train_label,
            missing=np.nan,
            feature_names=self.__application_train_feature.columns
        )
        self.__application_test_dmatrix = xgb.DMatrix(
            data=self.__application_test_feature,
            missing=np.nan,
            feature_names=self.__application_test_feature.columns
        )

    def model_fit(self):
        self.__params["tree_method"] = "hist"
        self.__clf = xgb.train(self.__params, self.__application_train_dmatrix, num_boost_round=300)

    def model_predict(self):
        # proba 出现负数
        self.__sample_submission["TARGET"] = np.clip(self.__clf.predict(self.__application_test_dmatrix), 0, 1)
        self.__sample_submission.to_csv(
            "C:\\Users\\puhui\\PycharmProjects\\Home_Credit_Default_Risk\\20180520\XgBoostBaseline\\sample_submission_one.csv",
            index=False
        )

    def model_plot(self):
        xgb.plot_importance(self.__clf)
        plt.show()


if __name__ == "__main__":
    xbb = XgBoostBaseline(path="C:\\Users\\puhui\\PycharmProjects\\Home_Credit_Default_Risk\\Data")
    xbb.data_prepare()
    xbb.model_fit()
    xbb.model_predict()
    xbb.model_plot()