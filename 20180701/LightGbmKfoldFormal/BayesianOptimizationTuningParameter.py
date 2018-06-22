# coding:utf-8

import os
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
np.random.seed(7)


class BayesianOptimizationTuningParameter(object):
    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__train = None
        self.__train_label = None
        self.__train_feature = None
        self.__encoder = None
        self.__categorical_columns = None

        # parameter tuning
        self.__gbm_bo = None
        self.__gp_params = {"alpha": 1e-5}

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_select_feature_df.csv"))
        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop("TARGET", axis=1)

        self.__encoder = TargetEncoder()
        self.__categorical_columns = self.__train_feature.select_dtypes("object").columns.tolist()
        self.__encoder.fit(self.__train_feature[self.__categorical_columns], self.__train_label)
        self.__train_feature[self.__categorical_columns] = (
            self.__encoder.transform(self.__train_feature[self.__categorical_columns])
        )

    def parameter_tuning(self):
        def __cv(n_estimators, learning_rate,
                 max_depth, num_leaves, min_split_gain, min_child_weight,
                 colsample_bytree, subsample,
                 reg_alpha, reg_lambda):
            val = cross_val_score(
                LGBMClassifier(
                    n_estimators=int(n_estimators),
                    learning_rate=learning_rate,
                    max_depth=int(max_depth),
                    num_leaves=int(num_leaves),
                    min_split_gain=min_split_gain,
                    min_child_weight=min_child_weight,
                    colsample_bytree=colsample_bytree,
                    subsample=subsample,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda
                ),
                self.__train_feature,
                self.__train_label,
                "roc_auc",
                cv=2
            ).mean()

            return val

        self.__gbm_bo = BayesianOptimization(
            __cv,
            {
                "n_estimators": (500, 10000),
                "learning_rate": (0.01, 0.05),
                "max_depth": (4, 10),
                "num_leaves": (10, 100),
                "min_split_gain": (0.1, 0.999),
                "min_child_weight": (10, 100),
                "colsample_bytree": (0.1, 0.999),
                "subsample": (0.1, 0.999),
                "reg_alpha": (0.1, 0.999),
                "reg_lambda": (0.1, 0.999)
            }
        )

        self.__gbm_bo.maximize(n_iter=10, ** self.__gp_params)
        print(self.__gbm_bo.res["max"]["max_params"]["n_estimators"])
        print(self.__gbm_bo.res["max"]["max_params"]["learning_rate"])
        print(self.__gbm_bo.res["max"]["max_params"]["max_depth"])
        print(self.__gbm_bo.res["max"]["max_params"]["num_leaves"])
        print(self.__gbm_bo.res["max"]["max_params"]["min_split_gain"])
        print(self.__gbm_bo.res["max"]["max_params"]["min_child_weight"])
        print(self.__gbm_bo.res["max"]["max_params"]["colsample_bytree"])
        print(self.__gbm_bo.res["max"]["max_params"]["subsample"])
        print(self.__gbm_bo.res["max"]["max_params"]["reg_alpha"])
        print(self.__gbm_bo.res["max"]["max_params"]["reg_lambda"])
