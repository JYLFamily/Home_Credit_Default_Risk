# coding:utf-8

import os
import gc
import sys
import importlib
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


class Feature(object):
    def __init__(self, *, input_path, output_path):
        self.__fill_value_transformer_array = importlib.import_module(
            "20180603.StackingReallyMax.FillValueTransformerArray"
        )
        self.__indicate_feature_transformer_array = importlib.import_module(
            "20180603.StackingReallyMax.IndicateFeatureTransformerArray"
        )
        self.__input_path = input_path
        self.__output_path = output_path

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None

        # model fit
        self.__rfc = None
        self.__etc = None
        self.__gbc = None
        self.__xbc = None
        self.__lr_l1 = None
        self.__lr_l2 = None
        self.__net = None
        self.__knc = None

        self.__rfc_pl = None
        self.__etc_pl = None
        self.__gbc_pl = None
        self.__xbc_pl = None
        self.__lr_l1_pl = None
        self.__lr_l2_pl = None
        self.__net_pl = None
        self.__knc_pl = None

        self.__clf = None
        self.__oof_train = None
        self.__oof_test = None

        # model_feature_output
        self.__feature_train = None
        self.__feature_test = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "first_layer_train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "first_layer_test.csv"))

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(["TARGET"], axis=1)
        self.__test_feature = self.__test

        del self.__train, self.__test
        gc.collect()

    def model_fit(self):
        self.__rfc = RandomForestClassifier(n_jobs=-1)
        self.__etc = ExtraTreesClassifier(n_jobs=-1)
        self.__gbc = GradientBoostingClassifier()
        self.__xbc = XGBClassifier(n_jobs=-1)
        self.__lr_l1 = LogisticRegression(penalty="l1")
        self.__lr_l2 = LogisticRegression(penalty="l2")
        self.__net = MLPClassifier()
        self.__knc = KNeighborsClassifier(n_jobs=-1)

        self.__rfc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__indicate_feature_transformer_array.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fill_value_transformer_array.FillValueTransformerArray(filling_values=-999.0)),
                ("Clf", self.__rfc)
            ]
        )
        self.__etc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__indicate_feature_transformer_array.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fill_value_transformer_array.FillValueTransformerArray(filling_values=-999.0)),
                ("Clf", self.__etc)
            ]
        )
        self.__gbc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__indicate_feature_transformer_array.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fill_value_transformer_array.FillValueTransformerArray(filling_values=-999.0)),
                ("Clf", self.__gbc)
            ]
        )
        self.__xbc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__indicate_feature_transformer_array.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fill_value_transformer_array.FillValueTransformerArray(filling_values=-999.0)),
                ("Clf", self.__xbc)
            ]
        )
        self.__lr_l1_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__indicate_feature_transformer_array.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fill_value_transformer_array.FillValueTransformerArray(filling_values=0)),
                ("Clf", self.__lr_l1)
            ]
        )
        self.__lr_l2_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__indicate_feature_transformer_array.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fill_value_transformer_array.FillValueTransformerArray(filling_values=0)),
                ("Clf", self.__lr_l2)
            ]
        )
        self.__net_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__indicate_feature_transformer_array.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fill_value_transformer_array.FillValueTransformerArray(filling_values=0)),
                ("Clf", self.__net)
            ]
        )
        self.__knc_pl = Pipeline(
            steps=[
                ("AddIndicator", self.__indicate_feature_transformer_array.IndicateFeatureTransformerArray(columns=[9, 10, 11])),
                ("FillNa", self.__fill_value_transformer_array.FillValueTransformerArray(filling_values=0)),
                ("Clf", self.__knc)
            ]
        )

        self.__clf = StackingCVClassifier(
            classifiers=[self.__rfc_pl,
                         self.__etc_pl,
                         self.__gbc_pl,
                         self.__xbc_pl,
                         self.__lr_l1_pl,
                         self.__lr_l2_pl,
                         self.__net_pl,
                         self.__knc_pl],
            meta_classifier=self.__lr_l1_pl,
            use_probas=True,
            cv=2,
            store_train_meta_features=True,
            verbose=True
        )

        self.__clf.fit(self.__train_feature.values, self.__train_label.values)
        self.__oof_train = self.__clf.train_meta_features_
        self.__oof_test = self.__clf.predict_meta_features(self.__test_feature.values)

        self.__oof_train = self.__oof_train[:, [i for i in range(2*len(self.__clf.classifiers)) if i % 2 != 0]]
        self.__oof_test = self.__oof_test[:, [i for i in range(2*len(self.__clf.classifiers)) if i % 2 != 0]]

        self.__oof_train = pd.DataFrame(
            self.__oof_train,
            columns=["rf_2", "et_2", "gb_2", "xg_2", "lr_l1_2", "lr_l2_2", "net_2", "knc_2"]
        )
        self.__oof_test = pd.DataFrame(
            self.__oof_test,
            columns=["rf_2", "et_2", "gb_2", "xg_2", "lr_l1_2", "lr_l2_2", "net_2", "knc_2"]
        )

    def model_feature_output(self):
        self.__feature_train = pd.concat([self.__train_feature, self.__oof_train], axis=1)
        self.__feature_test = pd.concat([self.__test_feature, self.__oof_test], axis=1)

        self.__feature_train.to_csv(os.path.join(self.__output_path, "feature_train_res.csv"), index=False)
        self.__feature_test.to_csv(os.path.join(self.__output_path, "feature_test_res.csv"), index=False)


if __name__ == "__main__":
    feature = Feature(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    feature.data_prepare()
    feature.model_fit()
    feature.model_feature_output()

