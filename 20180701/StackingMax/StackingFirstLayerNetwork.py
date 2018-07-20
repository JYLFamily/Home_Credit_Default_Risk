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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop
np.random.seed(7)


class StackingFirstLayerNetwork(object):

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
        self.__model_three_layers = None
        self.__model_four_layers = None
        self.__model_five_layers = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_select_feature_df.csv"), nrows=200)
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_select_feature_df.csv"), nrows=200)

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

                clf.compile(optimizer=RMSprop(), loss="binary_crossentropy", metrics=["accuracy"])
                clf.fit(trn_x.values, trn_y.values, epochs=10, batch_size=512)
                pred_val = clf.predict(val_x.values).reshape(-1, )  # keras 返回的 shape (num_sample, 1)
                pred_test = clf.predict(test_feature.values).reshape(-1, )

                oof_train[val_idx] = pred_val
                oof_test += pred_test / folds.n_splits

            return oof_train.reshape((-1, 1)), oof_test.reshape((-1, 1))

        self.__model_three_layers = Sequential([
            Dense(128, input_dim=self.__train_feature.shape[1], activation="relu"),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        self.__model_four_layers = Sequential([
            Dense(256, input_dim=self.__train_feature.shape[1], activation="relu"),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        self.__model_five_layers = Sequential([
            Dense(512, input_dim=self.__train_feature.shape[1], activation="relu"),
            Dropout(0.2),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])

        keras_three_layers_oof_train, keras_three_layers_oof_test = __get_oof(
            self.__model_three_layers,
            self.__train_feature,
            self.__train_label,
            self.__test_feature
        )
        print("keras three layers oof complete !")
        keras_four_layers_oof_train, keras_four_layers_oof_test = __get_oof(
            self.__model_four_layers,
            self.__train_feature,
            self.__train_label,
            self.__test_feature
        )
        print("keras four layers oof complete !")
        keras_five_layers_oof_train, keras_five_layers_oof_test = __get_oof(
            self.__model_five_layers,
            self.__train_feature,
            self.__train_label,
            self.__test_feature
        )
        print("keras five layers oof complete !")

        self.__oof_train = np.hstack((
            keras_three_layers_oof_train,
            keras_four_layers_oof_train,
            keras_five_layers_oof_train
        ))
        self.__oof_test = np.hstack((
            keras_three_layers_oof_test,
            keras_four_layers_oof_test,
            keras_five_layers_oof_test
        ))

    def model_predict(self):
        self.__oof_train = pd.DataFrame(
            self.__oof_train,
            columns=[
                "keras_three_layers", "keras_four_layers", "keras_fives_layers"
            ]
        )
        self.__oof_test = pd.DataFrame(
            self.__oof_test,
            columns=[
                "keras_three_layers", "keras_four_layers", "keras_fives_layers"
            ]
        )
        self.__first_layer_train = self.__oof_train
        self.__first_layer_test = self.__oof_test
        self.__first_layer_train.to_csv(os.path.join(self.__output_path, "first_layer_network_train.csv"), index=False)
        self.__first_layer_test.to_csv(os.path.join(self.__output_path, "first_layer_network_test.csv"), index=False)

if __name__ == "__main__":

    sm = StackingFirstLayerNetwork(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
        # input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\feature_data_V5",
        # output_path="D:\\Kaggle\\Home_Credit_Default_Risk\\feature_data_V5"
    )
    sm.data_prepare()
    sm.model_fit()
    sm.model_predict()


