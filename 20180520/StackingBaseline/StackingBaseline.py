# coding:utf-8

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from category_encoders import LeaveOneOutEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.classifier import StackingCVClassifier
np.random.seed(7)


class StackingBaseline(object):

    def __init__(self, *, path):
        self.__path = path
        self.__application_train = None
        self.__application_test = None
        self.__sample_submission = None

        # data prepare
        self.__application_train_feature = None
        self.__application_train_label = None
        self.__application_test_feature = None

        self.__categorical_columns = None
        self.__numeric_columns = None

        # numeric handle
        # categorical handle
        self.__encoder = None

        # model fit
        self.__lr = None
        self.__ef = None
        self.__rf = None
        self.__gb = None
        self.__xgb = None
        self.__sclf = None

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__path, "application_train.csv"))
        self.__application_test = pd.read_csv(os.path.join(self.__path, "application_test.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__path, "sample_submission.csv"))

        self.__application_train = self.__application_train.drop("SK_ID_CURR", axis=1)
        self.__application_test = self.__application_test.drop("SK_ID_CURR", axis=1)

        self.__application_train_feature = self.__application_train[[i for i in self.__application_train.columns if i != "TARGET"]]
        self.__application_train_label = self.__application_train["TARGET"]
        self.__application_test_feature = self.__application_test

        self.__categorical_columns = self.__application_train_feature.select_dtypes(include=["object"]).columns.tolist()
        self.__numeric_columns = [i for i in self.__application_train_feature.columns if i not in self.__categorical_columns]

    def numeric_handle(self):
        self.__application_train_feature[self.__numeric_columns] = self.__application_train_feature[self.__numeric_columns].fillna(-999.0)
        self.__application_test_feature[self.__numeric_columns] = self.__application_test_feature[self.__numeric_columns].fillna(-999.0)

    def categorical_handle(self):
        self.__application_train_feature[self.__categorical_columns] = (
            self.__application_train_feature[self.__categorical_columns].fillna("missing")
        )

        self.__encoder = LeaveOneOutEncoder()
        self.__encoder.fit(self.__application_train_feature[self.__categorical_columns], self.__application_train_label)
        self.__application_train_feature[self.__categorical_columns] = self.__encoder.transform(
            self.__application_train_feature[self.__categorical_columns]
        )
        self.__application_test_feature[self.__categorical_columns] = self.__encoder.transform(
            self.__application_test_feature[self.__categorical_columns]
        )

    def model_fit(self):
        self.__ef = ExtraTreesClassifier(n_jobs=-1)
        self.__rf = RandomForestClassifier(n_jobs=-1)
        self.__lr = LogisticRegression()
        self.__gb = GradientBoostingClassifier()
        self.__xgb = XGBClassifier(n_jobs=-1, missing=-999.0)
        self.__sclf = StackingCVClassifier(
            classifiers=[self.__ef, self.__rf, self.__gb, self.__xgb],
            meta_classifier=self.__lr,
            use_probas=True,
            cv=3
        )
        # sclf 需要的是 numpy array
        self.__sclf.fit(self.__application_train_feature.values, self.__application_train_label.values)

    def model_predict(self):
        self.__sample_submission["TARGET"] = np.clip(self.__sclf.predict_proba(self.__application_test_feature.values)[:, 1], 0, 1)
        self.__sample_submission.to_csv(
            "C:\\Users\\puhui\\PycharmProjects\\Home_Credit_Default_Risk\\20180520\StackingBaseline\\sample_submission2.csv",
            index=False
        )


if __name__ == "__main__":
    sb = StackingBaseline(path="C:\\Users\\puhui\\PycharmProjects\\Home_Credit_Default_Risk\\Data")
    sb.data_prepare()
    sb.numeric_handle()
    sb.categorical_handle()
    sb.model_fit()
    sb.model_predict()