# coding:utf-8

import os
import gc
import sys
import importlib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
np.random.seed(7)


class StackingSecondLayer(object):

    def __init__(self, *, input_path, output_path, output_file_name):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__output_file_name = output_file_name
        self.__fvt = importlib.import_module("FillValueTransformerArray")
        self.__ift = importlib.import_module("IndicateFeatureTransformerArray")

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__sample_submission = None
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None

        # model fit
        self.__rfc = None
        self.__etc = None
        self.__gbc = None
        self.__xbc = None
        self.__lr = None
        self.__mr = None
        self.__knc = None

        self.__rfc_pl = None
        self.__etc_pl = None
        self.__gbc_pl = None
        self.__xbc_pl = None
        self.__lr_pl = None
        self.__mr_pl = None
        self.__knc_pl = None

        self.__clf = None
        self.__oof_train = None
        self.__oof_test = None

        # model predict
        self.__meta = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "first_layer_train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "first_layer_test.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path, "sample_submission_one.csv"))

        self.__train_feature = self.__train.loc[:, [i for i in self.__train.columns if i != "TARGET"]]
        self.__train_label = self.__train.loc[:, ["TARGET"]].squeeze()
        del self.__train

        self.__test_feature = self.__test
        del self.__test

        gc.collect()

    def model_fit(self):
        self.__rfc = RandomForestClassifier(n_jobs=-1)
        self.__etc = ExtraTreesClassifier(n_jobs=-1)
        self.__gbc = GradientBoostingClassifier()
        self.__xbc = XGBClassifier(missing=-999.0, n_jobs=-1)
        self.__lr = LogisticRegression(penalty="l1")
        self.__mr = MLPClassifier()
        self.__knc = KNeighborsClassifier(n_jobs=-1)

        self.__rfc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__ift.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fvt.FillValueTransformerArray(filling_values=-999.0)),
                ("Clf", self.__rfc)
            ]
        )
        self.__etc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__ift.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fvt.FillValueTransformerArray(filling_values=-999.0)),
                ("Clf", self.__etc)
            ]
        )
        self.__gbc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__ift.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fvt.FillValueTransformerArray(filling_values=-999.0)),
                ("Clf", self.__gbc)
            ]
        )
        self.__xbc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__ift.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fvt.FillValueTransformerArray(filling_values=-999.0)),
                ("Clf", self.__xbc)
            ]
        )
        self.__lr_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__ift.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fvt.FillValueTransformerArray(filling_values=0)),
                ("Clf", self.__lr)
            ]
        )
        self.__mr_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__ift.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fvt.FillValueTransformerArray(filling_values=0)),
                ("Clf", self.__mr)
            ]
        )
        self.__knc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__ift.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fvt.FillValueTransformerArray(filling_values=0)),
                ("Clf", self.__knc)
            ]
        )

        self.__clf = StackingCVClassifier(
            classifiers=[self.__rfc_pl,
                         self.__etc_pl,
                         self.__gbc_pl,
                         self.__xbc_pl,
                         self.__lr_pl,
                         self.__mr_pl,
                         self.__knc_pl],
            meta_classifier=self.__xbc,
            use_probas=True,
            cv=5,
            store_train_meta_features=True,
            verbose=True
        )

        self.__clf.fit(self.__train_feature.values, self.__train_label.values)
        self.__oof_train = self.__clf.train_meta_features_
        self.__oof_test = self.__clf.predict_meta_features(self.__test_feature.values)

        # self.__oof_train = self.__oof_train[:, [i for i in range(len(self.__clf.classifiers)) if i % 2 != 0]]
        # self.__oof_test = self.__oof_test[:, [i for i in range(len(self.__clf.classifiers)) if i % 2 != 0]]

    def model_predict(self):
        print("------------------------------------")
        self.__meta = GridSearchCV(
            estimator=XGBClassifier(),
            param_grid={
                "max_depth": [1, 3],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [10, 30, 50, 70, 90]
            },
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        self.__meta.fit(self.__oof_train, self.__train_label)
        print(roc_auc_score(self.__train_label, self.__meta.predict_proba(self.__oof_train)[:, 1]))
        print(self.__meta.best_params_)
        print(self.__meta.best_score_)
        self.__sample_submission["TARGET"] = np.clip(self.__clf.predict_proba(self.__test_feature.values)[:, 1], 0, 1)
        self.__sample_submission.to_csv(os.path.join(self.__output_path, self.__output_file_name), index=False)


if __name__ == "__main__":
    ssl = StackingSecondLayer(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\feature_data_V3",
        output_path="D:\\Code\\Python\\Home_Credit_Default_Risk\\20180603\\StackingReallyMax",
        output_file_name="sample_submission_one.csv"
        # input_path=sys.argv[1],
        # output_path=sys.argv[2],
        # output_file_name="sample_submission_one.csv"
    )
    ssl.data_prepare()
    ssl.model_fit()
    ssl.model_predict()
