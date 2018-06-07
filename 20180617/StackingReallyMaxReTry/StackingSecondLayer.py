# coding:utf-8

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
np.random.seed(7)


class StackingSecondLayer(object):
    def __init__(self, *, input_path, output_path, output_file_name):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__output_file_name = output_file_name
        self.__sample_submission = None

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None

        # self.__scaler = None

        # model fit
        self.__params = None
        self.__dtrain = None
        self.__dtest = None
        self.__clf = None

        # model predict

    def data_prepare(self):
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path, "sample_submission.csv"))

        self.__train = pd.read_csv(os.path.join(self.__input_path, "first_layer_train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "first_layer_test.csv"))

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop("TARGET", axis=1)
        self.__train_feature["EXT_SOURCE_1_na_indicator"] = self.__train_feature["EXT_SOURCE_1"].isna().astype(int)
        self.__train_feature["EXT_SOURCE_2_na_indicator"] = self.__train_feature["EXT_SOURCE_2"].isna().astype(int)
        self.__train_feature["EXT_SOURCE_3_na_indicator"] = self.__train_feature["EXT_SOURCE_3"].isna().astype(int)

        self.__test_feature = self.__test
        self.__test_feature["EXT_SOURCE_1_na_indicator"] = self.__test_feature["EXT_SOURCE_1"].isna().astype(int)
        self.__test_feature["EXT_SOURCE_2_na_indicator"] = self.__test_feature["EXT_SOURCE_2"].isna().astype(int)
        self.__test_feature["EXT_SOURCE_3_na_indicator"] = self.__test_feature["EXT_SOURCE_3"].isna().astype(int)
        self.__test_feature = self.__test_feature[self.__train_feature.columns.tolist()]

        # 线性模型需要
        # self.__scaler = MinMaxScaler()
        # self.__scaler.fit(self.__train_feature)
        # self.__train_feature = self.__scaler.transform(self.__train_feature)
        # self.__test_feature = self.__scaler.transform(self.__test_feature)

    def model_fit(self):
        self.__params = {
            "objective": "reg:logistic",
            "booster": "gbtree",
            "tree_method": "hist",
            "max_depth": 3
        }
        self.__dtrain = xgb.DMatrix(
            data=self.__train_feature,
            label=self.__train_label,
            feature_names=self.__train_feature.columns
        )
        self.__dtest = xgb.DMatrix(
            data=self.__test_feature,
            feature_names=self.__test_feature.columns
        )
        self.__clf = xgb.train(self.__params, self.__dtrain, num_boost_round=10)

    def model_predict(self):
        print(roc_auc_score(self.__train_label, self.__clf.predict(self.__dtrain)))
        self.__sample_submission["TARGET"] = self.__clf.predict(self.__dtest)
        self.__sample_submission.to_csv(os.path.join(self.__output_path, self.__output_file_name), index=False)


if __name__ == "__main__":
    ssl = StackingSecondLayer(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\feature_data_V3",
        output_path="D:\\Code\\Python\\Home_Credit_Default_Risk\\20180617\\StackingReallyMaxReTry",
        output_file_name="sample_submission.csv"
    )
    ssl.data_prepare()
    ssl.model_fit()
    ssl.model_predict()