# coding:utf-8

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.utils import shuffle
from category_encoders import TargetEncoder
from sklearn.metrics import roc_auc_score
np.random.seed(7)


class CatBoostKfold(object):

    def __init__(self, *, input_path_1, input_path_2, output_path):
        self.__input_path_1 = input_path_1
        self.__input_path_2 = input_path_2
        self.__output_path = output_path

        self.__sample_submission = None
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_res, self.__test_res = [None for _ in range(2)]

        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None
        self.__categorical_index = None
        self.__encoder = None
        self.__numeric_index = None

        self.__folds = None
        self.__oof_preds = None
        self.__sub_preds = None
        self.__cat = None

    def data_prepare(self):
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path_1, "sample_submission.csv"))
        self.__train = pd.read_csv(os.path.join(self.__input_path_1, "train_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path_1, "test_feature_df.csv"))
        self.__train_res = pd.read_csv(os.path.join(self.__input_path_2, "feature_train_res.csv"))
        self.__test_res = pd.read_csv(os.path.join(self.__input_path_2, "feature_test_res.csv"))

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(["SK_ID_CURR", "TARGET"], axis=1)
        self.__test_feature = self.__test[self.__train_feature.columns]

        self.__train_res = self.__train_res.drop(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"], axis=1)
        self.__test_res = self.__test_res.drop(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"], axis=1)

        self.__train_feature = pd.concat([self.__train_feature, self.__train_res], axis=1)
        self.__test_feature = pd.concat([self.__test_feature, self.__test_res], axis=1)

        self.__categorical_index = np.where(self.__train_feature.dtypes == "object")[0]
        self.__train_feature.iloc[:, self.__categorical_index] = (
            self.__train_feature.iloc[:, self.__categorical_index].fillna("missing")
        )
        self.__test_feature.iloc[:, self.__categorical_index] = (
            self.__test_feature.iloc[:, self.__categorical_index].fillna("missing")
        )

        self.__encoder = TargetEncoder()
        self.__encoder.fit(self.__train_feature.iloc[:, self.__categorical_index], self.__train_label)
        self.__train_feature.iloc[:, self.__categorical_index] = (
            self.__encoder.transform(self.__train_feature.iloc[:, self.__categorical_index])
        )
        self.__test_feature.iloc[:, self.__categorical_index] = (
            self.__encoder.transform(self.__test_feature.iloc[:, self.__categorical_index])
        )

        # There are NaNs in test dataset (feature number 77) but there were no NaNs in learn dataset"
        self.__numeric_index = np.where(self.__train_feature.dtypes != "object")[0]
        self.__train_feature.iloc[:, self.__numeric_index] = (
            self.__train_feature.iloc[:, self.__numeric_index].apply(
                lambda x: x.fillna(-999999.0) if x.median() > 0 else x.fillna(999999.0)
            )
        )
        self.__test_feature.iloc[:, self.__numeric_index] = (
            self.__test_feature.iloc[:, self.__numeric_index].apply(
                lambda x: x.fillna(-999999.0) if x.median() > 0 else x.fillna(999999.0)
            )
        )

        # blending 之前需要 shuffle, 这里其实并不需要, 因为后面 StratifiedKFold shuffle
        self.__train_feature, self.__train_label = shuffle(self.__train_feature, self.__train_label)

    def model_fit(self):
        self.__folds = StratifiedKFold(n_splits=5, shuffle=True)
        self.__oof_preds = np.zeros(shape=self.__train_feature.shape[0])
        self.__sub_preds = np.zeros(shape=self.__test_feature.shape[0])

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(self.__train_feature, self.__train_label)):
            trn_x, trn_y = self.__train_feature.iloc[trn_idx], self.__train_label.iloc[trn_idx]
            val_x, val_y = self.__train_feature.iloc[val_idx], self.__train_label.iloc[val_idx]

            self.__cat = CatBoostClassifier(
                iterations=6000,
                od_wait=200,
                od_type="Iter",
                eval_metric="AUC"
            )
            self.__cat.fit(
                trn_x,
                trn_y,
                eval_set=[(val_x, val_y)],
                use_best_model=True
            )
            pred_val = self.__cat.predict_proba(val_x)[:, 1]
            pred_test = self.__cat.predict_proba(self.__test_feature)[:, 1]

            self.__oof_preds[val_idx] = pred_val
            self.__sub_preds += pred_test / self.__folds.n_splits
            print("Fold %2d AUC : %.6f" % (n_fold + 1, roc_auc_score(val_y, self.__oof_preds[val_idx])))
        print("Full AUC score %.6f" % roc_auc_score(self.__train_label, self.__oof_preds))

    def model_predict(self):
        self.__sample_submission["TARGET"] = self.__sub_preds
        self.__sample_submission.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    cbk = CatBoostKfold(
        input_path_1=sys.argv[1],
        input_path_2=sys.argv[2],
        output_path=sys.argv[3]
    )
    cbk.data_prepare()
    cbk.model_fit()
    cbk.model_predict()