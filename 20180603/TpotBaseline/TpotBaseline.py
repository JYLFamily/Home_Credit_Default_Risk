# coding:utf-8

import os
import sys
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import Imputer
from tpot import TPOTClassifier
np.random.seed(7)


class TpotBaseline(object):

    def __init__(self, *, input_path, output_path, output_file_name):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__output_file_name = output_file_name

        self.__train, self.__test = [None for _ in range(2)]
        self.__sample_submission = None
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None
        self.__categorical_index = None
        self.__numeric_index = None
        self.__encoder = None
        self.__imputer = None

        self.__clf = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_feature_df.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path, "sample_submission.csv"))

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop("TARGET", axis=1)
        self.__test_feature = self.__test[self.__train_feature.columns]

        # 离散变量缺失值处理 + 连续化
        self.__categorical_index = np.where(self.__train_feature.dtypes == "object")[0]
        self.__numeric_index = np.where(self.__train_feature.dtypes != "object")[0]

        self.__train_feature.iloc[:, self.__categorical_index] = (
            self.__train_feature.iloc[:, self.__categorical_index].fillna("missing")
        )
        self.__test_feature.iloc[:, self.__categorical_index] = (
            self.__test_feature.iloc[:, self.__categorical_index].fillna("missing")
        )
        self.__encoder = ce.TargetEncoder()
        self.__encoder.fit(
            self.__train_feature.iloc[:, self.__categorical_index],
            self.__train_label
        )
        self.__train_feature.iloc[:, self.__categorical_index] = self.__encoder.transform(
            self.__train_feature.iloc[:, self.__categorical_index]
        )
        self.__test_feature.iloc[:, self.__categorical_index] = self.__encoder.transform(
            self.__test_feature.iloc[:, self.__categorical_index]
        )

        # 连续变量缺失值处理
        self.__imputer = Imputer(strategy="median")
        self.__imputer.fit(
            self.__train_feature.iloc[:, self.__numeric_index]
        )
        self.__train_feature.iloc[:, self.__numeric_index] = self.__imputer.transform(
            self.__train_feature.iloc[:, self.__numeric_index]
        )
        self.__test_feature.iloc[:, self.__numeric_index] = self.__imputer.transform(
            self.__test_feature.iloc[:, self.__numeric_index]
        )

    def model_fit(self):
        self.__clf = TPOTClassifier(
            scoring="roc_auc",
            n_jobs=-1,
            verbosity=2
        )
        self.__clf.fit(self.__train_feature.values, self.__train_label.values)

    def model_predict(self):
        self.__sample_submission["TARGET"] = self.__clf.predict_proba(self.__test_feature.values)[:, 1]
        self.__sample_submission.to_csv(os.path.join(self.__output_path, self.__output_file_name), index=False)
        self.__clf.export(os.path.join(self.__output_path, "tpot_baseline.py"))


if __name__ == "__main__":
    tb = TpotBaseline(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
        output_file_name=sys.argv[3]
    )
    tb.data_prepare()
    tb.model_fit()
    tb.model_predict()