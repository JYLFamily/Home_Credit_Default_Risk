# coding:utf-8

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from category_encoders import TargetEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif
np.random.seed(7)


class FeatureSelection(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path = input_path
        self.__output_path = output_path

        # data prepare
        self.__train = None
        self.__train_feature = None
        self.__train_label = None
        self.__numeric_columns = None
        self.__categorical_columns = None
        self.__imputer = None
        self.__encoder = None
        self.__unsupervise_selector = None
        self.__supervise_selector = None

        # data output
        self.__columns = None
        self.__columns_rename = None
        self.__columns_mapping = None

        self.__train_feature_label_df_fs = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train_feature_df.csv"))
        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop(["SK_ID_CURR", "TARGET"], axis=1)
        self.__numeric_columns = self.__train_feature.select_dtypes(exclude="object").columns.tolist()
        self.__categorical_columns = self.__train_feature.select_dtypes(include="object").columns.tolist()

        self.__imputer = Imputer(strategy="median")
        self.__imputer.fit(self.__train_feature[self.__numeric_columns])
        self.__train_feature[self.__numeric_columns] = (
            self.__imputer.transform(self.__train_feature[self.__numeric_columns])
        )

        self.__train_feature[self.__categorical_columns] = (
            self.__train_feature[self.__categorical_columns].fillna("missing")
        )
        self.__encoder = TargetEncoder()
        self.__encoder.fit(self.__train_feature[self.__categorical_columns], self.__train_label)
        self.__train_feature[self.__categorical_columns] = (
            self.__encoder.transform(self.__train_feature[self.__categorical_columns])
        )

        # 非监督 feature filter
        self.__unsupervise_selector = VarianceThreshold()
        self.__unsupervise_selector.fit(self.__train_feature)
        self.__train_feature = (
            pd.DataFrame(
                self.__unsupervise_selector.transform(self.__train_feature),
                columns=[i for i, j in zip(self.__train_feature.columns, self.__unsupervise_selector.get_support()) if j == 1]
            )
        )

        # 监督 feature filter
        pd.concat([
            pd.Series(
                self.__train_feature.columns
            ).to_frame("feature"),

            pd.Series(
                mutual_info_classif(self.__train_feature, self.__train_label)
            ).to_frame("mi")
        ], axis=1).to_csv(os.path.join(self.__output_path, "train_feature_df_fs_mi.csv"), index=False)

        # self.__supervise_selector = SelectPercentile(mutual_info_classif, 90)
        # self.__supervise_selector.fit(self.__train_feature, self.__train_label)
        # self.__train_feature = (
        #     pd.DataFrame(
        #         self.__supervise_selector.transform(self.__train_feature),
        #         columns=[i for i, j in zip(self.__train_feature.columns, self.__supervise_selector.get_support()) if j == 1]
        #     )
        # )

    def data_output(self):
        self.__columns = pd.Series(self.__train_feature.columns)
        self.__columns_rename = pd.Series(self.__train_feature.columns).apply(
            lambda x: x.replace("(", "_").replace(")", "_").replace(".", "_")
        )
        self.__columns_mapping = (
            pd.concat([self.__columns.to_frame("columns"), self.__columns_rename.to_frame("columns_rename")], axis=1)
        )
        self.__columns_mapping.to_csv(os.path.join(self.__output_path, "columns_mapping.csv"), index=False)

        self.__train_feature.columns = self.__columns_rename
        self.__train_feature_label_df_fs = pd.concat([self.__train_feature, self.__train_label.to_frame("TARGET")], axis=1)
        self.__train_feature_label_df_fs.to_csv(os.path.join(self.__output_path, "train_feature_label_df_fs.csv"), index=False)


if __name__ == "__main__":
    fs = FeatureSelection(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    fs.data_prepare()