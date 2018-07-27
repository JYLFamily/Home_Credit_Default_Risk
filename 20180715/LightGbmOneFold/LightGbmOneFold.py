# coding:utf-8

import re
import os
import sys
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
np.random.seed(7)


class LightGbmOneFold(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # data prepare
        self.__sample_submission = None
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label = None
        self.__categorical_columns = None
        self.__encoder = None

        # model fit
        self.__folds = None
        self.__train_preds = None
        self.__test_preds = None
        self.__gbm = None

    def data_prepare(self):
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path, "sample_submission.csv"))

        # selected feature
        self.__train = pd.read_csv(
            os.path.join(self.__input_path, "train_select_feature_df.csv"))
        self.__test = pd.read_csv(
            os.path.join(self.__input_path, "test_select_feature_df.csv"))

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(
            ["TARGET"] + [col for col in self.__train.columns.tolist() if re.search(r"SK_ID", col)], axis=1)
        self.__test_feature = self.__test[self.__train_feature.columns.tolist()]

        self.__categorical_columns = self.__train_feature.select_dtypes("object").columns.tolist()
        self.__encoder = TargetEncoder()
        self.__encoder.fit(self.__train_feature.loc[:, self.__categorical_columns], self.__train_label)
        self.__train_feature.loc[:, self.__categorical_columns] = (
            self.__encoder.transform(self.__train_feature.loc[:, self.__categorical_columns])
        )
        self.__test_feature.loc[:, self.__categorical_columns] = (
            self.__encoder.transform(self.__test_feature.loc[:, self.__categorical_columns])
        )

    def model_fit(self):
        feature_importance_df = pd.DataFrame()

        self.__gbm = LGBMClassifier(
            n_estimators=5000,
            learning_rate=0.0128,
            max_depth=8,
            num_leaves=11,
            min_split_gain=0.0018,
            min_child_weight=2.6880,
            colsample_bytree=0.5672,
            subsample=0.6406,
            reg_alpha=3.5025,
            reg_lambda=0.9549,
            n_jobs=-1
        )

        self.__gbm.fit(self.__train_feature, self.__train_label, verbose=True)
        self.__train_preds = self.__gbm.predict_proba(self.__train_feature)[:, 1]
        self.__test_preds = self.__gbm.predict_proba(self.__test_feature)[:, 1]

        feature_importance_df["feature"] = pd.Series(self.__train_feature.columns)
        feature_importance_df["importance"] = self.__gbm.feature_importances_
        feature_importance_df.to_csv(os.path.join(self.__output_path, "feature_importance.csv"), index=False)
        print("Train AUC score %.6f" % roc_auc_score(self.__train_label, self.__train_preds))

    def model_predict(self):
        self.__sample_submission["TARGET"] = self.__test_preds
        self.__sample_submission.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    lgof = LightGbmOneFold(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    lgof.data_prepare()
    lgof.model_fit()
    lgof.model_predict()