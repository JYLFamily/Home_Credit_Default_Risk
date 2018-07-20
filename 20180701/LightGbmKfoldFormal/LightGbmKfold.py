# coding:utf-8

import re
import os
import sys
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
np.random.seed(8)


class LightGbmKfold(object):
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
        self.__oof_preds = None
        self.__sub_preds = None
        self.__gbm = None
        # self.__metric_weight = []

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

    def model_fit(self):
        self.__folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=8)
        self.__oof_preds = np.zeros(shape=self.__train_feature.shape[0])
        self.__sub_preds = np.zeros(shape=self.__test_feature.shape[0])
        # self.__sub_preds = np.zeros(shape=(self.__test_feature.shape[0], 5))

        feature_importance_df = pd.DataFrame()
        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(self.__train_feature, self.__train_label)):
            trn_x, trn_y = self.__train_feature.iloc[trn_idx], self.__train_label.iloc[trn_idx]
            val_x, val_y = self.__train_feature.iloc[val_idx], self.__train_label.iloc[val_idx]

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

            self.__gbm.fit(
                trn_x,
                trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric="auc",
                verbose=True,
                early_stopping_rounds=5
            )
            pred_val = self.__gbm.predict_proba(val_x, num_iteration=self.__gbm.best_iteration_)[:, 1]
            pred_test = self.__gbm.predict_proba(self.__test_feature, num_iteration=self.__gbm.best_iteration_)[:, 1]

            self.__oof_preds[val_idx] = pred_val
            self.__sub_preds += pred_test / self.__folds.n_splits
            # self.__sub_preds[:, n_fold] = pred_test

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = pd.Series(self.__train_feature.columns)
            fold_importance_df["importance"] = self.__gbm.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            # 保存 weight
            # self.__metric_weight.append(roc_auc_score(val_y, self.__oof_preds[val_idx]))
            print("Fold %2d AUC : %.6f" % (n_fold + 1, roc_auc_score(val_y, self.__oof_preds[val_idx])))

        feature_importance_df.to_csv(os.path.join(self.__output_path, "feature_importance.csv"), index=False)
        print("Full AUC score %.6f" % roc_auc_score(self.__train_label, self.__oof_preds))

    def model_predict(self):
        # weight sum
        # self.__metric_weight = pd.Series(self.__metric_weight).rank()
        # self.__metric_weight = self.__metric_weight / self.__metric_weight.sum()
        # self.__metric_weight = self.__metric_weight.values.reshape((5, 1))
        # self.__sub_preds = np.dot(self.__sub_preds, self.__metric_weight)
        self.__sample_submission["TARGET"] = self.__sub_preds
        self.__sample_submission.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    lgk = LightGbmKfold(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    lgk.data_prepare()
    lgk.model_fit()
    lgk.model_predict()



