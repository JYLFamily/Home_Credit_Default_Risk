# coding:utf-8

import re
import os
import sys
import importlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
np.random.seed(7)


class StackingFirstLayerLinear(object):

    def __init__(self, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path
        self.__skwp = importlib.import_module("SklWrapper")

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature = None

        self.__categorical_index = None
        self.__numeric_index = None

        # filler encoder scaler
        self.__filler, self.__encoder, self.__scaler = [None for _ in range(3)]
        self.__oof_train, self.__oof_test = [None for _ in range(2)]
        self.__first_layer_train, self.__first_layer_test = [None for _ in range(2)]

        # model fit

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_select_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_select_feature_df.csv"))

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(
            ["TARGET"] + [col for col in self.__train.columns.tolist() if re.search(r"SK_ID", col)], axis=1)
        self.__test_feature = self.__test[self.__train_feature.columns.tolist()]

        # drop column na
        self.__train_feature = self.__train_feature[
            list((self.__train_feature.isna().sum() / self.__train_feature.isna().count())[
                     (self.__train_feature.isna().sum() / self.__train_feature.isna().count()) < 0.2].index)
        ]
        self.__test_feature = self.__test_feature[self.__train_feature.columns.tolist()]

        # columns 而不是 index
        self.__categorical_index = self.__train_feature.select_dtypes(include="object").columns.tolist()
        self.__numeric_index = self.__train_feature.select_dtypes(exclude="object").columns.tolist()

        # filler Imputer all np.nan remove column
        self.__filler = Imputer(strategy="median")
        self.__filler.fit(self.__train_feature[self.__numeric_index])
        self.__train_feature[self.__numeric_index] = self.__filler.transform(
            self.__train_feature[self.__numeric_index]
        )
        self.__test_feature[self.__numeric_index] = self.__filler.transform(
            self.__test_feature[self.__numeric_index]
        )

        # encoder
        self.__encoder = TargetEncoder()
        self.__encoder.fit(self.__train_feature[self.__categorical_index], self.__train_label)
        self.__train_feature[self.__categorical_index] = self.__encoder.transform(
            self.__train_feature[self.__categorical_index]
        )
        self.__test_feature[self.__categorical_index] = self.__encoder.transform(
            self.__test_feature[self.__categorical_index]
        )

        # scaler pandas in numpy out
        self.__scaler = MinMaxScaler()
        self.__scaler.fit(self.__train_feature)
        self.__train_feature = pd.DataFrame(
            self.__scaler.transform(self.__train_feature),
            columns=self.__train_feature.columns
        )
        self.__test_feature = pd.DataFrame(
            self.__scaler.transform(self.__test_feature),
            columns=self.__test_feature.columns
        )

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

        lr_p1 = self.__skwp.SklWrapper(
            clf=LogisticRegression,
            init_params={
                "penalty": "l1"
            }
        )
        lr_p2 = self.__skwp.SklWrapper(
            clf=LogisticRegression,
            init_params={
                "penalty": "l2"
            }
        )
        mlp_unit_100 = self.__skwp.SklWrapper(
            clf=MLPClassifier,
            init_params={
                "hidden_layer_sizes": (100, )
            }
        )
        mlp_unit_200 = self.__skwp.SklWrapper(
            clf=MLPClassifier,
            init_params={
                "hidden_layer_sizes": (200, )
            }
        )
        mlp_unit_300 = self.__skwp.SklWrapper(
            clf=MLPClassifier,
            init_params={
                "hidden_layer_sizes": (300, )
            }
        )
        # mlp_unit_5_100 = self.__skwp.SklWrapper(
        #     clf=MLPClassifier,
        #     init_params={
        #         "hidden_layer_sizes": (100, 100, 100, 100, 100)
        #     }
        # )
        # mlp_unit_5_300 = self.__skwp.SklWrapper(
        #     clf=MLPClassifier,
        #     init_params={
        #         "hidden_layer_sizes": (300, 300, 300, 300, 300)
        #     }
        # )
        # mlp_unit_5_900 = self.__skwp.SklWrapper(
        #     clf=MLPClassifier,
        #     init_params={
        #         "hidden_layer_sizes": (900, 900, 900, 900, 900)
        #     }
        # )

        lr_p1_oof_train, lr_p1_oof_test = __get_oof(
            lr_p1,
            self.__train_feature,
            self.__train_label,
            self.__test_feature
        )
        print("lr l1 oof complete !")
        lr_p2_oof_train, lr_p2_oof_test = __get_oof(
            lr_p2,
            self.__train_feature,
            self.__train_label,
            self.__test_feature
        )
        print("lr l2 oof complete !")
        mlp_unit_100_oof_train, mlp_unit_100_oof_test = __get_oof(
            mlp_unit_100,
            self.__train_feature,
            self.__train_label,
            self.__test_feature
        )
        print("mlp 100 oof complete !")
        mlp_unit_200_oof_train, mlp_unit_200_oof_test = __get_oof(
            mlp_unit_200,
            self.__train_feature,
            self.__train_label,
            self.__test_feature
        )
        print("mlp 200 oof complete !")
        mlp_unit_300_oof_train, mlp_unit_300_oof_test = __get_oof(
            mlp_unit_300,
            self.__train_feature,
            self.__train_label,
            self.__test_feature
        )
        print("mlp 300 oof complete !")
        # mlp_unit_5_100_oof_train, mlp_unit_5_100_oof_test = __get_oof(
        #     mlp_unit_5_100,
        #     self.__train_feature,
        #     self.__train_label,
        #     self.__test_feature
        # )
        # print("mlp 5 100 oof complete !")
        # mlp_unit_5_300_oof_train, mlp_unit_5_300_oof_test = __get_oof(
        #     mlp_unit_5_300,
        #     self.__train_feature,
        #     self.__train_label,
        #     self.__test_feature
        # )
        # print("mlp 5 300 oof complete !")
        # mlp_unit_5_900_oof_train, mlp_unit_5_900_oof_test = __get_oof(
        #     mlp_unit_5_900,
        #     self.__train_feature,
        #     self.__train_label,
        #     self.__test_feature
        # )
        # print("mlp 5 900 oof complete !")

        self.__oof_train = np.hstack((
            lr_p1_oof_train,
            lr_p2_oof_train,
            mlp_unit_100_oof_train,
            mlp_unit_200_oof_train,
            mlp_unit_300_oof_train
        ))
        self.__oof_test = np.hstack((
            lr_p1_oof_test,
            lr_p2_oof_test,
            mlp_unit_100_oof_test,
            mlp_unit_200_oof_test,
            mlp_unit_300_oof_test
        ))

    def model_predict(self):
        self.__oof_train = pd.DataFrame(
            self.__oof_train,
            columns=[
                "lr_p1", "lr_p2", "mlp_unit_100", "mlp_unit_200", "mlp_unit_300"
            ]
        )
        self.__oof_test = pd.DataFrame(
            self.__oof_test,
            columns=[
                "lr_p1", "lr_p2", "mlp_unit_100", "mlp_unit_200", "mlp_unit_300"
            ]
        )
        self.__first_layer_train = self.__oof_train
        self.__first_layer_test = self.__oof_test
        self.__first_layer_train.to_csv(os.path.join(self.__output_path, "first_layer_linear_train.csv"), index=False)
        self.__first_layer_test.to_csv(os.path.join(self.__output_path, "first_layer_linear_test.csv"), index=False)

if __name__ == "__main__":

    sm = StackingFirstLayerLinear(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    sm.data_prepare()
    sm.model_fit()
    sm.model_predict()