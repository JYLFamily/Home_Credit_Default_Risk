# coding:utf-8

import re
import os
import sys
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
np.random.seed(7)


class BayesianOptimizationGbdt(object):
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
        self.__gbm_params = None
        self.__gp_params = {"alpha": 1e-4}

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_select_feature_df.csv"))
        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(
            ["TARGET"] + [col for col in self.__train.columns.tolist() if re.search(r"SK_ID", col)], axis=1)

        self.__encoder = TargetEncoder()
        self.__categorical_columns = self.__train_feature.select_dtypes("object").columns.tolist()
        self.__encoder.fit(self.__train_feature[self.__categorical_columns], self.__train_label)
        self.__train_feature[self.__categorical_columns] = (
            self.__encoder.transform(self.__train_feature[self.__categorical_columns])
        )

    def parameter_tuning(self):
        def __cv(
                n_estimators, learning_rate,
                max_depth, num_leaves, min_split_gain, min_child_weight,
                colsample_bytree, subsample, reg_alpha, reg_lambda):
            val = cross_val_score(
                LGBMClassifier(
                    n_estimators=max(int(round(n_estimators)), 1),
                    learning_rate=max(min(learning_rate, 1.0), 0),
                    max_depth=max(int(round(max_depth)), 1),
                    # 如果 num_leaves > 2 ^ round(max_depth) 时 leaf-wise 的树就会太深导致 overfitting
                    num_leaves=max(2 ^ int(round(max_depth)) if num_leaves > 2 ^ int(round(max_depth)) else int(round(num_leaves)), 1),
                    min_split_gain=max(min_split_gain, 0),
                    min_child_weight=max(min_child_weight, 0),
                    colsample_bytree=max(min(colsample_bytree, 1.0), 0),
                    subsample=max(min(subsample, 1.0), 0),
                    reg_alpha=max(reg_alpha, 0),
                    reg_lambda=max(reg_lambda, 0),
                    n_jobs=-1,
                    verbose=-1
                ),
                self.__train_feature,
                self.__train_label,
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
            ).mean()

            return val

        self.__gbm_params = {
            # Gradient boosting parameter
            "n_estimators": (5000, 7000),
            "learning_rate": (0.001, 0.03),
            # tree parameter
            "max_depth": (4, 10),
            "num_leaves": (10, 200),
            "min_split_gain": (0.00001, 0.1),
            "min_child_weight": (1, 100),
            # bagging parameter
            "colsample_bytree": (0.5, 1.0),
            "subsample": (0.5, 1.0),
            # reg parameter
            "reg_alpha": (0, 10),
            "reg_lambda": (0, 10)
        }
        self.__gbm_bo = BayesianOptimization(__cv, self.__gbm_params)
        self.__gbm_bo.maximize(init_points=10,  n_iter=50, ** self.__gp_params)


if __name__ == "__main__":
    botp = BayesianOptimizationGbdt(
        input_path=sys.argv[1]
    )
    botp.data_prepare()
    botp.parameter_tuning()