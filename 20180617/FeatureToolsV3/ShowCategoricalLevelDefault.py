# coding:utf-8

import os
import re
import pandas as pd


class ShowCategoricalLevelDefault(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path = input_path
        self.__output_path = output_path

        # data prepare
        self.__application_train = None
        self.__bureau = None
        self.__previous_application = None

        self.__application_train_categorical = None
        self.__bureau_categorical = None
        self.__previous_application_categorical = None

        # show categorical level
        self.__level_mean = None
        self.__level_count = None

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__input_path, "application_train.csv"))
        self.__bureau = pd.read_csv(os.path.join(self.__input_path, "bureau.csv"))
        self.__previous_application = pd.read_csv(os.path.join(self.__input_path, "previous_application.csv"))

        # categorical feature
        self.__bureau_categorical = self.__bureau.select_dtypes(include="object").columns.tolist()
        self.__bureau_categorical.extend(
            [col for col in self.__bureau.columns.tolist() if re.search(r"FLAG", col)]
        )

        self.__previous_application_categorical = self.__previous_application.select_dtypes(include="object").columns.tolist()
        self.__previous_application_categorical.extend(
            [col for col in self.__previous_application.columns.tolist() if re.search(r"FLAG", col)]
        )

    def show_categorical_level(self):
        self.__level_mean = pd.DataFrame
        self.__level_count = pd.DataFrame

        self.__bureau[self.__bureau_categorical] = self.__bureau[self.__bureau_categorical].fillna("missing")
        for col in self.__bureau_categorical:

            temp = self.__bureau[["SK_ID_CURR", col]].groupby("SK_ID_CURR")[col].agg(
                lambda x: x.value_counts().index[0]
            ).to_frame("MODE_" + col)

            self.__application_train = self.__application_train[["SK_ID_CURR", "TARGET"]].merge(
                temp,
                left_on=["SK_ID_CURR"],
                right_index=True,
                how="left"
            )

            print("********************************")
            print(col)
            print(self.__application_train.groupby("MODE_" + col)["TARGET"].mean().to_frame().merge(
                self.__application_train.groupby("MODE_" + col)["TARGET"].count().to_frame(),
                left_index=True,
                right_index=True
            ))

        self.__previous_application[self.__previous_application_categorical] = (
            self.__previous_application[self.__previous_application_categorical].fillna("missing")
        )
        for col in self.__previous_application_categorical:
            temp = self.__previous_application[["SK_ID_CURR", col]].groupby("SK_ID_CURR")[col].agg(
                lambda x: x.value_counts().index[0]
            ).to_frame("MODE_" + col)

            self.__application_train = self.__application_train[["SK_ID_CURR", "TARGET"]].merge(
                temp,
                left_on=["SK_ID_CURR"],
                right_index=True,
                how="left"
            )

            print("********************************")
            print(col)
            print(self.__application_train.groupby("MODE_" + col)["TARGET"].mean().to_frame().merge(
                self.__application_train.groupby("MODE_" + col)["TARGET"].count().to_frame(),
                left_index=True,
                right_index=True
            ))


if __name__ == "__main__":
    scld = ShowCategoricalLevelDefault(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data",
        output_path=None
    )
    scld.data_prepare()
    scld.show_categorical_level()