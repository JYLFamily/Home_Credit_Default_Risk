# coding:utf-8

import os
import gc
import sys
import importlib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
np.random.seed(7)


class JamesStepherd(object):
    def __init__(self, input_path, output_path):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__AddManualFeature = importlib.import_module("AddManualFeature")

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label = None

        # manual feature
        self.__add_manual_feature = None

        # categorical feature & numeric feature
        self.__categorical_index = None
        self.__numeric_index = None

        # model fit
        self.__folds = None
        self.__oof_preds = None
        self.__sub_preds = None
        self.__cat = None

        # model predict
        self.__sample_submission = None

    def data_prepare(self):
        # raw feature
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_feature_df.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path, "sample_submission.csv"))

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(["SK_ID_CURR", "TARGET"], axis=1)
        self.__test_feature = self.__test[self.__train_feature.columns.tolist()]

        del self.__train, self.__test
        gc.collect()

        # add manual feature
        self.__add_manual_feature = self.__AddManualFeature.AddManualFeature(
            train_feature=self.__train_feature, test_feature=self.__test_feature
        )
        self.__train_feature, self.__test_feature = self.__add_manual_feature.add_manual_feature()
        self.__test_feature = self.__test_feature[self.__train_feature.columns]

        # categorical feature
        self.__categorical_index = np.where(self.__train_feature.dtypes == "object")[0]
        self.__train_feature.iloc[:, self.__categorical_index] = (
            self.__train_feature.iloc[:, self.__categorical_index].fillna("missing")
        )
        self.__test_feature.iloc[:, self.__categorical_index] = (
            self.__test_feature.iloc[:, self.__categorical_index].fillna("missing")
        )

        # numeric feature
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

        # blending shuffle
        self.__train_feature, self.__train_label = shuffle(self.__train_feature, self.__train_label)

    def model_fit(self):
        self.__folds = StratifiedKFold(n_splits=5, shuffle=True)
        self.__oof_preds = np.zeros(shape=self.__train_feature.shape[0])
        self.__sub_preds = np.zeros(shape=self.__test_feature.shape[0])

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(self.__train_feature, self.__train_label)):
            trn_x, trn_y = self.__train_feature.iloc[trn_idx], self.__train_label.iloc[trn_idx]
            val_x, val_y = self.__train_feature.iloc[val_idx], self.__train_label.iloc[val_idx]

            self.__cat = CatBoostClassifier(
                depth=4,
                l2_leaf_reg=300,
                iterations=3000,
                od_wait=300,
                od_type="Iter",
                eval_metric="AUC"
            )
            self.__cat.fit(
                trn_x,
                trn_y,
                cat_features=self.__categorical_index,
                eval_set=[(val_x, val_y)],
                use_best_model=True
            )
            pred_val = self.__cat.predict_proba(val_x)[:, 1]
            pred_test = self.__cat.predict_proba(self.__test_feature)[:, 1]

            self.__oof_preds[val_idx] = pred_val
            self.__sub_preds += pred_test / self.__folds.n_splits

            pd.concat([
                pd.Series(self.__train_feature.columns).to_frame("feature_name"),
                pd.Series(self.__cat.feature_importances_).to_frame("feature_importances")
            ], axis=1).to_csv(os.path.join(self.__output_path, "feature_importance_" + str(n_fold) + ".csv"), index=False)

    def model_predict(self):
        self.__sample_submission["TARGET"] = self.__sub_preds
        self.__sample_submission.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    js = JamesStepherd(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    js.data_prepare()
    js.model_fit()
    js.model_predict()