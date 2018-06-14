# coding:utf-8

import os
import sys
import importlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
np.random.seed(7)


class StackingMax(object):

    def __init__(self, *, input_path, output_path, output_file_name):
        # import
        self.__cbdp = importlib.import_module("CatBoostDataPrepare")
        self.__xbdp = importlib.import_module("XgBoostDataPrepare")
        self.__skdp = importlib.import_module("SklDataPrepare")
        self.__cbwp = importlib.import_module("CatWrapper")
        self.__xbwp = importlib.import_module("XgbWrapper")
        self.__skwp = importlib.import_module("SklWrapper")

        # init
        self.__input_path = input_path
        self.__output_path = output_path
        self.__output_file_name = output_file_name

        # data prepare
        self.__train, self.__test, self.__sample_submission = [None for _ in range(3)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None

        self.__categorical_index = None
        self.__numeric_index = None

        self.__cat_train_feature, self.__cat_test_feature = [None for _ in range(2)]
        self.__xgb_train_feature, self.__xgb_test_feature = [None for _ in range(2)]
        self.__skl_train_feature, self.__skl_test_feature = [None for _ in range(2)]

        # model fit
        # 第一层 stacking
        self.__oof_train = None
        self.__oof_test = None
        # 第二层 stacking
        self.__sclf = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_feature_df.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path, "sample_submission_one.csv"))

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop("TARGET", axis=1)
        self.__test_feature = self.__test[self.__train_feature.columns]

        self.__categorical_index = np.where(self.__train_feature.dtypes == "object")[0]
        self.__numeric_index = np.where(self.__train_feature.dtypes != "object")[0]

        self.__cat_train_feature, self.__cat_test_feature = self.__cbdp.CatBoostDataPrepare(
            train_feature=self.__train_feature,
            test_feature=self.__test_feature
        ).data_prepare()

        self.__xgb_train_feature, self.__xgb_test_feature = self.__xbdp.XgBoostDataPrepare(
            train_feature=self.__train_feature,
            train_label=self.__train_label,
            test_feature=self.__test_feature
        ).data_prepare()

        self.__skl_train_feature, self.__skl_test_feature = self.__skdp.SklDataPrepare(
            train_feature=self.__train_feature,
            train_label=self.__train_label,
            test_feature=self.__test_feature
        ).data_prepare()

    def model_fit(self):
        # stacking 1
        def __get_oof(clf, train_feature, train_label, test_feature):
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
            oof_train = np.zeros(shape=train_feature.shape[0])
            oof_test = np.zeros(shape=test_feature.shape[0])  

            for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_feature, train_label)):
                trn_x, trn_y = train_feature.iloc[trn_idx], train_label.iloc[trn_idx]
                val_x, val_y = train_feature.iloc[val_idx], train_label.iloc[val_idx]

                clf.train(
                    trn_x,
                    trn_y
                )
                pred_val = clf.predict(val_x)
                pred_test = clf.predict(test_feature)

                oof_train[val_idx] = pred_val
                oof_test += pred_test / folds.n_splits

            return oof_train.reshape((-1, 1)), oof_test.reshape((-1, 1))

        cb = self.__cbwp.CatWrapper(
            clf=CatBoostClassifier,
            init_params={
                "iterations": 2900
            },
            train_params={  # 这里想到一个问题, stacking 是否就不能够使用 early_stopping 了
                "cat_features": self.__categorical_index,
            }
        )
        xg = self.__xbwp.XgbWrapper(
            clf=xgb,
            train_params={
                "tree_method": "hist",
                "num_boost_round": 2900
            }
        )
        et = self.__skwp.SklWrapper(
            clf=ExtraTreesClassifier,
            init_params={
                "n_estimators": 200,
                "n_jobs": -1
            }
        )
        rf = self.__skwp.SklWrapper(
            clf=RandomForestClassifier,
            init_params={
                "n_estimators": 200,
                "n_jobs": -1
            }
        )
        ad = self.__skwp.SklWrapper(
            clf=AdaBoostClassifier,
            init_params={
                "n_estimators": 1400
            }
        )

        cb_oof_train, cb_oof_test = __get_oof(
            cb,
            self.__cat_train_feature,
            self.__train_label,
            self.__cat_test_feature
        )
        print("catboost oof complete !")
        xg_oof_train, xg_oof_test = __get_oof(
            xg,
            self.__xgb_train_feature,
            self.__train_label,
            self.__xgb_test_feature
        )
        print("xgboost oof complete !")
        et_oof_train, et_oof_test = __get_oof(
            et,
            self.__skl_train_feature,
            self.__train_label,
            self.__skl_test_feature
        )
        print("et oof complete !")
        rf_oof_train, rf_oof_test = __get_oof(
            rf,
            self.__skl_train_feature,
            self.__train_label,
            self.__skl_test_feature
        )
        print("rf oof complete !")
        ad_oof_train, ad_oof_test = __get_oof(
            ad,
            self.__skl_train_feature,
            self.__train_label,
            self.__skl_test_feature
        )
        print("ad oof complete !")

        self.__oof_train = np.hstack((cb_oof_train, xg_oof_train, et_oof_train, rf_oof_train, ad_oof_train))
        self.__oof_test = np.hstack((cb_oof_test, xg_oof_test, et_oof_test, rf_oof_test, ad_oof_test))

        # stacking 2
        self.__sclf = StackingCVClassifier(
            classifiers=(
                [LogisticRegression(),
                 KNeighborsClassifier(),
                 GradientBoostingClassifier(),
                 RandomForestClassifier(),
                 MLPClassifier()]
            ),
            meta_classifier=LogisticRegression(),
            use_probas=True,
            cv=3
        )
        self.__sclf.fit(self.__oof_train, self.__train_label.values) # oof_train 是 Array
        print("stacking 2 complete !")

    def model_predict(self):
        self.__sample_submission["TARGET"] = np.clip(self.__sclf.predict_proba(self.__oof_test)[:, 1], 0, 1)
        self.__sample_submission.to_csv(os.path.join(self.__output_path, self.__output_file_name), index=False)


if __name__ == "__main__":

    sm = StackingMax(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
        output_file_name=sys.argv[3]
    )
    sm.data_prepare()
    sm.model_fit()
    sm.model_predict()