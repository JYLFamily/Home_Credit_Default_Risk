# coding:utf-8

import re
import os
import gc
import numpy as np
import pandas as pd


class JamesStepherd(object):
    def __init__(self, input_path, output_path):
        self.__input_path = input_path
        self.__output_path = output_path

        # data prepare
        self.__application_train = None
        self.__application_test = None
        self.__bureau = None
        self.__bureau_balance = None
        self.__credit_card_balance = None
        self.__installments_payments = None
        self.__pos_cash_balance = None
        self.__previous_application = None

        self.__application_train_TARGET = None

        # manual feature

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__input_path, "application_train.csv"))
        self.__application_test = pd.read_csv(os.path.join(self.__input_path, "application_test.csv"))
        self.__bureau = pd.read_csv(os.path.join(self.__input_path, "bureau.csv"))
        self.__bureau_balance = pd.read_csv(os.path.join(self.__input_path, "bureau_balance.csv"))
        # self.__credit_card_balance = pd.read_csv(os.path.join(self.__input_path, "credit_card_balance.csv"))
        # self.__installments_payments = pd.read_csv(os.path.join(self.__input_path, "installments_payments.csv"))
        # self.__pos_cash_balance = pd.read_csv(os.path.join(self.__input_path, "POS_CASH_balance.csv"))
        # self.__previous_application = pd.read_csv(os.path.join(self.__input_path, "previous_application.csv"))

        # handle application train and application test
        # self.__application_train_TARGET = self.__application_train["TARGET"]
        # self.__application_train = self.__application_train.drop("TARGET", axis=1)
        # self.__application_test = self.__application_test[self.__application_train.columns]

        # handle bureau and bureau_balance
        # bureau_balance agg bureau
        self.__bureau_balance = (
            pd.get_dummies(self.__bureau_balance, prefix="bureau_balance", dummy_na=True, columns=["STATUS"])
        )
        aggregations = dict()
        aggregations["MONTHS_BALANCE"] = ["size"]  # 一个还款周期的长度
        for col in self.__bureau_balance.columns:
            if re.search(r"bureau_balance", col):
                aggregations[col] = ["mean"]  # 一个还款周期各种还款状态的占比
        self.__bureau_balance = self.__bureau_balance.groupby("SK_ID_BUREAU").agg(aggregations)
        self.__bureau_balance.columns = (
            [i[0] + "_" + i[1] for i in self.__bureau_balance.columns.tolist()]
        )
        self.__bureau_balance = self.__bureau_balance.reset_index()
        self.__bureau = (
            self.__bureau.merge(self.__bureau_balance, left_on=["SK_ID_BUREAU"], right_on=["SK_ID_BUREAU"], how="left")
        )
        self.__bureau = self.__bureau.drop("SK_ID_BUREAU", axis=1)

        del self.__bureau_balance
        gc.collect()
        # bureau agg application train
        # bureau_balance agg application test

    def model_fit(self):
        pass

    def model_predict(self):
        pass


if __name__ == "__main__":
    js = JamesStepherd(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data",
        output_path=None
    )
    js.data_prepare()