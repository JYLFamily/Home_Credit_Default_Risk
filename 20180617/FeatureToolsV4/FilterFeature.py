# coding:utf-8

import os
import sys
import pandas as pd


class FilterFeature(object):
    def __init__(self, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path
        self.__train_feature_df, self.__test_feature_df = [None for _ in range(2)]

    def data_prepare(self):
        self.__train_feature_df = pd.read_csv(os.path.join(self.__input_path, "train_feature_df.csv"), nrows=10)
        self.__test_feature_df = pd.read_csv(os.path.join(self.__input_path, "test_feature_df.csv"))

    def data_output(self):
        self.__test_feature_df = (
            self.__test_feature_df[[col for col in self.__test_feature_df.columns if col in self.__train_feature_df.columns]]
        )

        self.__test_feature_df.to_csv(os.path.join(self.__output_path, "test_feature_df.csv"), index=False)

if __name__ == "__main__":
    ff = FilterFeature(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    ff.data_prepare()
    ff.data_output()
