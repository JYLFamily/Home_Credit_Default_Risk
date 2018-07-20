# coding:utf-8

import re
import os
import sys
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.utils import shuffle
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
np.random.seed(8)


class LightGbmOneFold(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # data prepare
        self.__sample_submission = None
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_feature_stacking_tree, self.__test_feature_stacking_tree = [None for _ in range(2)]
        self.__train_feature_stacking_linear, self.__test_feature_stacking_linear = [None for _ in range(2)]
        self.__train_feature_stacking_network, self.__test_feature_stacking_network = [None for _ in range(2)]
        self.__train_feature_stacking_gp, self.__test_feature_stacking_gp = [None for _ in range(2)]
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
        # stacking tree
        self.__train_feature_stacking_tree = pd.read_csv(
            os.path.join(self.__input_path, "first_layer_tree_train.csv"))
        self.__test_feature_stacking_tree = pd.read_csv(
            os.path.join(self.__input_path, "first_layer_tree_test.csv"))
        # stacking linear
        self.__train_feature_stacking_linear = pd.read_csv(
            os.path.join(self.__input_path, "first_layer_linear_train.csv"))
        self.__test_feature_stacking_linear = pd.read_csv(
            os.path.join(self.__input_path, "first_layer_linear_test.csv"))
        # stacking network
        self.__train_feature_stacking_network = pd.read_csv(
            os.path.join(self.__input_path, "first_layer_network_train.csv"))
        self.__test_feature_stacking_network = pd.read_csv(
            os.path.join(self.__input_path, "first_layer_network_test.csv"))
        # gp
        self.__train_feature_stacking_gp = pd.read_csv(
            os.path.join(self.__input_path, "genetic_train_feature.csv"))
        self.__test_feature_stacking_gp = pd.read_csv(
            os.path.join(self.__input_path, "genetic_test_feature.csv"))

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

        self.__train_feature = pd.concat(
            [self.__train_feature,
             self.__train_feature_stacking_tree,
             self.__train_feature_stacking_linear,
             self.__train_feature_stacking_network,
             self.__train_feature_stacking_gp], axis=1
        )
        self.__test_feature = pd.concat(
            [self.__test_feature,
             self.__test_feature_stacking_tree,
             self.__test_feature_stacking_linear,
             self.__test_feature_stacking_network,
             self.__test_feature_stacking_gp], axis=1
        )

        self.__train_feature, self.__train_label = shuffle(self.__train_feature, self.__train_label)

    def model_fit(self):
        feature_importance_df = pd.DataFrame()

        self.__gbm = LGBMClassifier(
            colsample_bytree=0.6659,
            learning_rate=0.0197,
            max_depth=8,
            min_child_weight=1.0652,
            min_split_gain=0.058,
            n_estimators=501,
            num_leaves=11,
            reg_alpha=2.2487,
            reg_lambda=6.2587,
            subsample=0.9401
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