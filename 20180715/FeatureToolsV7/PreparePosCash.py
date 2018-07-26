# coding:utf-8

import os
import pandas as pd


class PreparePosCash(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__pos_cash = None

        # data transform
        self.__start_time = pd.Timestamp("2018-07-20")

    def data_prepare(self):
        self.__pos_cash = pd.read_csv(os.path.join(self.__input_path, "POS_CASH_balance.csv"), nrows=10)
        self.__pos_cash = self.__pos_cash.drop(["SK_ID_CURR"], axis=1)

    def data_transform(self):
        self.__pos_cash["TIME_MONTHS_BALANCE"] = pd.to_timedelta(self.__pos_cash["MONTHS_BALANCE"], "M")

        self.__pos_cash["TIME_MONTHS_BALANCE"] += self.__start_time

        # 方便后续 featuretools 制定 variable types
        for col in self.__pos_cash.columns.tolist():
            if col in self.__pos_cash.select_dtypes(include="object").columns.tolist():
                self.__pos_cash.rename(columns={col: "FLAG_POS_CASH_" + col}, inplace=True)

        self.__pos_cash = pd.get_dummies(
            data=self.__pos_cash,
            dummy_na=True,
            columns=self.__pos_cash.select_dtypes(include="object").columns.tolist()
        )

    def data_generate(self):
        pass

    def data_return(self):
        # print(self.__pos_cash.shape)
        self.__pos_cash.to_csv(os.path.join(self.__input_path, "pos_cash_temp.csv"), index=False)

        return self.__pos_cash


if __name__ == "__main__":
    ppc = PreparePosCash(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data"
    )
    ppc.data_prepare()
    ppc.data_transform()
    ppc.data_generate()
    ppc.data_return()