# coding:utf-8

import os
import pandas as pd


class PrepareBureauBalance(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__bureau_balance = None

        # data transform
        self.__start_time = pd.Timestamp("2018-07-20")

    def data_prepare(self):
        self.__bureau_balance = pd.read_csv(os.path.join(self.__input_path, "bureau_balance.csv"), nrows=10)

    def data_transform(self):
        self.__bureau_balance["TIME_MONTHS_BALANCE"] = pd.to_timedelta(self.__bureau_balance["MONTHS_BALANCE"], "M")

        self.__bureau_balance["TIME_MONTHS_BALANCE"] += self.__start_time

        # 方便后续 featuretools 制定 variable types
        for col in self.__bureau_balance.columns.tolist():
            if col in self.__bureau_balance.select_dtypes(include="object").columns.tolist():
                self.__bureau_balance.rename(columns={col: "FLAG_BUREAU_BALANCE_" + col}, inplace=True)

        self.__bureau_balance = pd.get_dummies(
            data=self.__bureau_balance,
            prefix="FLAG_BUREAU_BALANCE",
            dummy_na=True,
            columns=self.__bureau_balance.select_dtypes(include="object").columns.tolist()
        )

    def data_generate(self):
        pass

    def data_return(self):
        # print(self.__bureau_balance.shape)
        # self.__bureau_balance.to_csv(os.path.join(self.__input_path, "bureau_balance_temp.csv"), index=False)

        return self.__bureau_balance


if __name__ == "__main__":
    pbb = PrepareBureauBalance(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data"
    )
    pbb.data_prepare()
    pbb.data_transform()
    pbb.data_generate()
    pbb.data_return()