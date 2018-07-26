# coding:utf-8

import os
import numpy as np
import pandas as pd


class PreparePreviousApplication(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__previous_application = None

        # data transform
        self.__start_time = pd.Timestamp("2018-07-20")

    def data_prepare(self):
        self.__previous_application = pd.read_csv(os.path.join(self.__input_path, "previous_application.csv"), nrows=10)

    def data_transform(self):
        # self.__previous_application = self.__previous_application.replace(365243.0, np.nan)
        self.__previous_application["TIME_DAYS_DECISION"] = pd.to_timedelta(self.__previous_application["DAYS_DECISION"], "D")
        self.__previous_application["TIME_DAYS_FIRST_DRAWING"] = pd.to_timedelta(self.__previous_application["DAYS_FIRST_DRAWING"], "D")
        self.__previous_application["TIME_DAYS_FIRST_DUE"] = pd.to_timedelta(self.__previous_application["DAYS_FIRST_DUE"], "D")
        self.__previous_application["TIME_DAYS_LAST_DUE_1ST_VERSION"] = pd.to_timedelta(self.__previous_application["DAYS_LAST_DUE_1ST_VERSION"], "D")
        self.__previous_application["TIME_DAYS_LAST_DUE"] = pd.to_timedelta(self.__previous_application["DAYS_LAST_DUE"], "D")
        self.__previous_application["TIME_DAYS_TERMINATION"] = pd.to_timedelta(self.__previous_application["DAYS_TERMINATION"], "D")

        self.__previous_application["TIME_DAYS_DECISION"] += self.__start_time
        self.__previous_application["TIME_DAYS_FIRST_DRAWING"] += self.__start_time
        self.__previous_application["TIME_DAYS_FIRST_DUE"] += self.__start_time
        self.__previous_application["TIME_DAYS_LAST_DUE_1ST_VERSION"] += self.__start_time
        self.__previous_application["TIME_DAYS_LAST_DUE"] += self.__start_time
        self.__previous_application["TIME_DAYS_TERMINATION"] += self.__start_time

        # 方便后续 featuretools 制定 variable types
        for col in self.__previous_application.columns.tolist():
            if col in self.__previous_application.select_dtypes(include="object").columns.tolist():
                self.__previous_application.rename(columns={col: "FLAG_PREVIOUS_APPLICATION_" + col}, inplace=True)

        self.__previous_application = pd.get_dummies(
            data=self.__previous_application,
            dummy_na=True,
            columns=self.__previous_application.select_dtypes(include="object").columns.tolist()
        )

    def data_generate(self):
        # 授信额度可能大于申请额度
        self.__previous_application["NEW_AMT_APPLICATION_DIVIDE_AMT_CREDIT"] = (
            # 申请额度 /
            # 授信额度
            self.__previous_application["AMT_APPLICATION"] /
            self.__previous_application["AMT_CREDIT"].replace(0, np.nan)
        )

        # 预付额度小于授信额度
        self.__previous_application["NEW_AMT_DOWN_PAYMENT_DIVIDE_AMT_CREDIT"] = (
            # 预付额度 /
            # 授信额度
            self.__previous_application["AMT_DOWN_PAYMENT"] /
            self.__previous_application["AMT_CREDIT"].replace(0, np.nan)
        )
        self.__previous_application["NEW_AMT_DOWN_PAYMENT_DIVIDE_AMT_CREDIT"] = (
            self.__previous_application["NEW_AMT_DOWN_PAYMENT_DIVIDE_AMT_CREDIT"].apply(lambda x: np.clip(x, 0, 1))
        )

        self.__previous_application["NEW_AMT_CREDIT_DIVIDE_AMT_ANNUITY"] = (
            self.__previous_application["AMT_CREDIT"] /
            self.__previous_application["AMT_ANNUITY"].replace(0, np.nan)
        )
        self.__previous_application["NEW_AMT_CREDIT_DIVIDE_AMT_GOODS_PRICE"] = (
            self.__previous_application["AMT_CREDIT"] /
            self.__previous_application["AMT_GOODS_PRICE"].replace(0, np.nan)
        )

    def data_return(self):
        # print(self.__previous_application.shape)
        # self.__previous_application.to_csv(os.path.join(self.__input_path, "previous_application_temp.csv"), index=False)

        return self.__previous_application


if __name__ == "__main__":
    ppa = PreparePreviousApplication(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data"
    )
    ppa.data_prepare()
    ppa.data_transform()
    ppa.data_generate()
    ppa.data_return()