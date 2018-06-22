# coding:utf-8

import os
import sys
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
np.random.seed(7)


class FeatureSelectionUseVariance(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path = input_path
        self.__output_path = output_path

        self.__train, self.__test = [None for _ in range(2)]
        self.__train_label = None
        self.__train_feature = None

        self.__categorical_columns = None
        self.__encoder = None

        self.__remove_feature = []

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_feature_df.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test_feature_df.csv"))
        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(["SK_ID_CURR", "TARGET"], axis=1)

        # encoder
        self.__categorical_columns = (
            self.__train_feature.select_dtypes(include="object").columns.tolist()
        )
        self.__train_feature[self.__categorical_columns] = (
            self.__train_feature[self.__categorical_columns].fillna("missing")
        )
        self.__encoder = TargetEncoder()
        self.__encoder.fit(self.__train_feature[self.__categorical_columns], self.__train_label)
        self.__train_feature[self.__categorical_columns] = (
            self.__encoder.transform(self.__train_feature[self.__categorical_columns])
        )

        for col in self.__train_feature.columns.tolist():
            if self.__train_feature.columns[col].std() == 0.:
                self.__remove_feature.append(col)

    def data_output(self):
        self.__train[[col for col in self.__train if col not in self.__remove_feature]].to_csv(
            os.path.join(self.__output_path, "train_select_feature_df.csv"),
            index=False
        )
        #
        self.__test[[col for col in self.__test if col not in self.__remove_feature]].to_csv(
            os.path.join(self.__output_path, "test_select_feature_df.csv"),
            index=False
        )


if __name__ == "__main__":
    fsuv = FeatureSelectionUseVariance(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    fsuv.data_prepare()
    fsuv.data_output()


