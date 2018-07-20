# coding:utf-8

import re
import os
import sys
import importlib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
np.random.seed(7)


class StackingFirstLayerTree(object):

    def __init__(self, *, input_path, output_path):
        # import
        self.__cbdp = importlib.import_module("CatboostDataPrepare")
        self.__lbdp = importlib.import_module("LgboostDataPrepare")
        self.__xbdp = importlib.import_module("XgboostDataPrepare")
        self.__skdp = importlib.import_module("SklDataPrepare")
        self.__cbwp = importlib.import_module("CatWrapper")
        self.__lbwp = importlib.import_module("LgbWrapper")
        self.__xbwp = importlib.import_module("XgbWrapper")
        self.__skwp = importlib.import_module("SklWrapper")

        # init
        self.__input_path = input_path
        self.__output_path = output_path

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None

        self.__categorical_index = None
        self.__numeric_index = None

        self.__cat_train_feature, self.__cat_test_feature = [None for _ in range(2)]
        self.__lgb_train_feature, self.__lgb_test_feature = [None for _ in range(2)]
        self.__xgb_train_feature, self.__xgb_test_feature = [None for _ in range(2)]
        self.__skl_train_feature, self.__skl_test_feature = [None for _ in range(2)]

        self.__oof_train, self.__oof_test = [None for _ in range(2)]
        # self.__ext_source_train, self.__ext_source_test = [None for _ in range(2)]

        # data output
        self.__first_layer_train, self.__first_layer_test = [None for _ in range(2)]

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_select_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_select_feature_df.csv"))

        # self.__train_label = self.__train["TARGET"]
        # self.__train_feature = self.__train.drop(["TARGET", "SK_ID_CURR"], axis=1)
        # self.__test = self.__test.drop("SK_ID_CURR", axis=1)
        # self.__test_feature = self.__test[self.__train_feature.columns]
        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(
            ["TARGET"] + [col for col in self.__train.columns.tolist() if re.search(r"SK_ID", col)], axis=1)
        self.__test_feature = self.__test[self.__train_feature.columns.tolist()]

        # 使用 stacking 前要 shuffle 防止 KFold 的时候使用到未来的信息
        # shuffle 函数 Pandas in Pandas Out, 所以不用担心
        # self.__train_feature, self.__train_label = shuffle(self.__train_feature, self.__train_label)

        # 保留 EXT_SOURCE 信息
        # self.__ext_source_train = self.__train_feature[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        # self.__ext_source_test = self.__test_feature[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        # self.__train_feature = self.__train_feature.drop(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"], axis=1)
        # self.__test_feature = self.__test_feature.drop(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"], axis=1)

        self.__categorical_index = np.where(self.__train_feature.dtypes == "object")[0]
        self.__numeric_index = np.where(self.__train_feature.dtypes != "object")[0]

        self.__cat_train_feature, self.__cat_test_feature = self.__cbdp.CatboostDataPrepare(
            train_feature=self.__train_feature,
            test_feature=self.__test_feature
        ).data_prepare()

        self.__lgb_train_feature, self.__lgb_test_feature = self.__lbdp.LgboostDataPrepare(
            train_feature=self.__train_feature,
            test_feature=self.__test_feature
        ).data_prepare()

        self.__xgb_train_feature, self.__xgb_test_feature = self.__xbdp.XgboostDataPrepare(
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

        # model
        cb = self.__cbwp.CatWrapper(
            clf=CatBoostClassifier,
            init_params={
                "iterations": 3000
            },
            train_params={  # 这里想到一个问题, stacking 是否就不能够使用 early_stopping 了
                "cat_features": self.__categorical_index
            }
        )
        lb_gbdt = self.__lbwp.LgbWrapper(
            clf=lgb,
            dataset_params={
                "categorical_feature": self.__categorical_index
            },
            train_params={
                "objective": "binary",
                "num_boost_round": 3000,
                "feature_fraction": 0.95,
                "bagging_fraction": 0.90
            }
        )
        lb_goss = self.__lbwp.LgbWrapper(
            clf=lgb,
            dataset_params={
                "categorical_feature": self.__categorical_index
            },
            train_params={
                "objective": "binary",
                "boosting": "goss",
                "num_boost_round": 3000
            }
        )
        lb_dart = self.__lbwp.LgbWrapper(
            clf=lgb,
            dataset_params={
                "categorical_feature": self.__categorical_index
            },
            train_params={
                "objective": "binary",
                "boosting": "dart",
                "num_boost_round": 3000
            }
        )
        xg_dart = self.__xbwp.XgbWrapper(
            clf=xgb,
            train_params={
                "objective": "reg:logistic",
                "booster": "dart",
                "num_boost_round": 3000
            }
        )
        xg_tree = self.__xbwp.XgbWrapper(
            clf=xgb,
            train_params={
                "objective": "reg:logistic",
                "booster": "gbtree",
                "tree_method": "hist",
                "num_boost_round": 3000
            }
        )
        xg_linear = self.__xbwp.XgbWrapper(
            clf=xgb,
            train_params={
                "objective": "reg:logistic",
                "booster": "gblinear",
                "num_boost_round": 3000
            }
        )
        et_gini = self.__skwp.SklWrapper(
            clf=ExtraTreesClassifier,
            init_params={
                "criterion": "gini",
                "n_estimators": 300,
                "n_jobs": -1
            }
        )
        et_entropy = self.__skwp.SklWrapper(
            clf=ExtraTreesClassifier,
            init_params={
                "criterion": "entropy",
                "n_estimators": 300,
                "n_jobs": -1
            }
        )
        rf_gini = self.__skwp.SklWrapper(
            clf=RandomForestClassifier,
            init_params={
                "criterion": "gini",
                "n_estimators": 300,
                "n_jobs": -1
            }
        )
        rf_entropy = self.__skwp.SklWrapper(
            clf=RandomForestClassifier,
            init_params={
                "criterion": "entropy",
                "n_estimators": 1,
                "n_jobs": -1
            }
        )

        # get oof
        cb_oof_train, cb_oof_test = __get_oof(
            cb,
            self.__cat_train_feature,
            self.__train_label,
            self.__cat_test_feature
        )
        print("catboost oof complete !")
        lb_gbdt_oof_train, lb_gbdt_oof_test = __get_oof(
            lb_gbdt,
            self.__lgb_train_feature,
            self.__train_label,
            self.__lgb_test_feature
        )
        print("lightgbm gbdt oof complete !")
        lb_goss_oof_train, lb_goss_oof_test = __get_oof(
            lb_goss,
            self.__lgb_train_feature,
            self.__train_label,
            self.__lgb_test_feature
        )
        print("lightgbm goss oof complete !")
        lb_dart_oof_train, lb_dart_oof_test = __get_oof(
            lb_dart,
            self.__lgb_train_feature,
            self.__train_label,
            self.__lgb_test_feature
        )
        print("lightgbm dart oof complete !")
        xg_dart_oof_train, xg_dart_oof_test = __get_oof(
            xg_dart,
            self.__xgb_train_feature,
            self.__train_label,
            self.__xgb_test_feature
        )
        print("xgboost dart oof complete !")
        xg_tree_oof_train, xg_tree_oof_test = __get_oof(
            xg_tree,
            self.__xgb_train_feature,
            self.__train_label,
            self.__xgb_test_feature
        )
        print("xgboost tree oof complete !")
        xg_linear_oof_train, xg_linear_oof_test = __get_oof(
            xg_linear,
            self.__xgb_train_feature,
            self.__train_label,
            self.__xgb_test_feature
        )
        print("xgboost linear oof complete !")
        et_gini_oof_train, et_gini_oof_test = __get_oof(
            et_gini,
            self.__skl_train_feature,
            self.__train_label,
            self.__skl_test_feature
        )
        print("et gini oof complete !")
        et_entropy_oof_train, et_entropy_oof_test = __get_oof(
            et_entropy,
            self.__skl_train_feature,
            self.__train_label,
            self.__skl_test_feature
        )
        print("et entropy oof complete !")
        rf_gini_oof_train, rf_gini_oof_test = __get_oof(
            rf_gini,
            self.__skl_train_feature,
            self.__train_label,
            self.__skl_test_feature
        )
        print("rf gini oof complete !")
        rf_entropy_oof_train, rf_entropy_oof_test = __get_oof(
            rf_entropy,
            self.__skl_train_feature,
            self.__train_label,
            self.__skl_test_feature
        )
        print("rf entropy oof complete !")

        self.__oof_train = np.hstack((
            cb_oof_train,
            lb_gbdt_oof_train,
            lb_goss_oof_train,
            lb_dart_oof_train,
            xg_dart_oof_train,
            xg_tree_oof_train,
            xg_linear_oof_train,
            et_gini_oof_train,
            et_entropy_oof_train,
            rf_gini_oof_train,
            rf_entropy_oof_train
        ))
        self.__oof_test = np.hstack((
            cb_oof_test,
            lb_gbdt_oof_test,
            lb_goss_oof_test,
            lb_dart_oof_test,
            xg_dart_oof_test,
            xg_tree_oof_test,
            xg_linear_oof_test,
            et_gini_oof_test,
            et_entropy_oof_test,
            rf_gini_oof_test,
            rf_entropy_oof_test
        ))

    def model_predict(self):
        self.__oof_train = pd.DataFrame(
            self.__oof_train,
            columns=[
                "cb",
                "lb_gbdt", "lb_goss", "lb_dart",
                "xg_dart", "xg_tree", "xg_linear",
                "et_gini", "et_entropy", "rf_gini", "rf_entropy"]
        )
        self.__oof_test = pd.DataFrame(
            self.__oof_test,
            columns=[
                "cb",
                "lb_gbdt", "lb_goss", "lb_dart",
                "xg_dart", "xg_tree", "xg_linear",
                "et_gini", "et_entropy", "rf_gini", "rf_entropy"]
        )
        self.__first_layer_train = self.__oof_train
        self.__first_layer_test = self.__oof_test
        self.__first_layer_train.to_csv(os.path.join(self.__output_path, "first_layer_tree_train.csv"), index=False)
        self.__first_layer_test.to_csv(os.path.join(self.__output_path, "first_layer_tree_test.csv"), index=False)


if __name__ == "__main__":

    sm = StackingFirstLayerTree(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    sm.data_prepare()
    sm.model_fit()
    sm.model_predict()