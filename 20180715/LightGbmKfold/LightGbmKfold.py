# coding:utf-8

import re
import os
import gc
import sys
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
np.random.seed(7)


class LightGbmKfold(object):
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

        del self.__train, self.__test, self.__categorical_columns, self.__encoder
        gc.collect()

    def model_fit(self):
        self.__folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
        self.__oof_preds = np.zeros(shape=self.__train_feature.shape[0])
        self.__sub_preds = np.zeros(shape=self.__test_feature.shape[0])
        # self.__sub_preds = np.zeros(shape=(self.__test_feature.shape[0], 5))

        feature_importance_df = pd.DataFrame()
        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(self.__train_feature, self.__train_label)):
            trn_x, trn_y = self.__train_feature.iloc[trn_idx], self.__train_label.iloc[trn_idx]
            val_x, val_y = self.__train_feature.iloc[val_idx], self.__train_label.iloc[val_idx]

            self.__gbm = LGBMClassifier(
                n_estimators=8989,
                learning_rate=0.0137,
                max_depth=8,
                num_leaves=200,
                min_split_gain=0.0320,
                min_child_weight=1.3459,
                colsample_bytree=0.5128,
                subsample=0.8438,
                reg_alpha=8.0363,
                reg_lambda=2.6900,
                n_jobs=-1
            )

            self.__gbm.fit(trn_x, trn_y)
            pred_val = self.__gbm.predict_proba(val_x)[:, 1]
            pred_test = self.__gbm.predict_proba(self.__test_feature)[:, 1]

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

            del trn_x, trn_y, val_x, val_y
            gc.collect()

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



