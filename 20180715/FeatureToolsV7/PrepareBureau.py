# coding:utf-8

import os
import numpy as np
import pandas as pd


def correct(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x


class PrepareBureau(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__bureau = None

        # data transform
        self.__start_time = pd.Timestamp("2018-07-20")

    def data_prepare(self):
        self.__bureau = pd.read_csv(os.path.join(self.__input_path, "bureau.csv"), nrows=10)

    def data_transform(self):
        # self.__bureau = self.__bureau.replace(365243.0, np.nan)
        self.__bureau["DAYS_CREDIT"] = pd.to_timedelta(self.__bureau["DAYS_CREDIT"], "D")
        self.__bureau["DAYS_CREDIT_ENDDATE"] = pd.to_timedelta(self.__bureau["DAYS_CREDIT_ENDDATE"], "D")
        self.__bureau["DAYS_ENDDATE_FACT"] = pd.to_timedelta(self.__bureau["DAYS_ENDDATE_FACT"], "D")
        self.__bureau["DAYS_CREDIT_UPDATE"] = pd.to_timedelta(self.__bureau["DAYS_CREDIT_UPDATE"], "D")

        self.__bureau["DAYS_CREDIT"] += self.__start_time
        self.__bureau["DAYS_CREDIT_ENDDATE"] += self.__start_time
        self.__bureau["DAYS_ENDDATE_FACT"] += self.__start_time
        self.__bureau["DAYS_CREDIT_UPDATE"] += self.__start_time

        # 方便后续 featuretools 制定 variable types
        for col in self.__bureau.columns.tolist():
            if col in self.__bureau.select_dtypes(include="object").columns.tolist():
                self.__bureau.rename(columns={col: "FLAG_BUREAU_" + col}, inplace=True)

        self.__bureau = pd.get_dummies(
            data=self.__bureau,
            dummy_na=True,
            columns=self.__bureau.select_dtypes(include="object").columns.tolist()
        )

    def data_generate(self):
        self.__bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_CREDIT_SUM"] = (
            # 信用局的当前债务 /
            # 信用局的当前信用额度
            self.__bureau["AMT_CREDIT_SUM_DEBT"] / self.__bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_CREDIT_SUM"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_CREDIT_SUM"].apply(lambda x: correct(x))
        )

        self.__bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_CREDIT_SUM"] = (
            # 信用局报告的信用卡当前额度 /
            # 信用局的当前信用额度
            self.__bureau["AMT_CREDIT_SUM_LIMIT"] / self.__bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_CREDIT_SUM"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_CREDIT_SUM"].apply(lambda x: correct(x))
        )

        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"] = (
            # 信用局的逾期总额 /
            # 信用局的当前信用额度
            self.__bureau["AMT_CREDIT_SUM_OVERDUE"] / self.__bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"].apply(lambda x: correct(x))
        )

        self.__bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_ANNUITY"] = (
            # 信用局的当前债务 /
            # 信用局的年金
            self.__bureau["AMT_CREDIT_SUM_DEBT"] / self.__bureau["AMT_ANNUITY"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_ANNUITY"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_ANNUITY"].apply(lambda x: correct(x))
        )

        self.__bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_ANNUITY"] = (
            # 信用局报告的信用卡当前额度 /
            # 信用局的年金
            self.__bureau["AMT_CREDIT_SUM_LIMIT"] / self.__bureau["AMT_ANNUITY"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_ANNUITY"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_ANNUITY"].apply(lambda x: correct(x))
        )

        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_ANNUITY"] = (
            # 信用局的逾期总额 /
            # 信用局的年金
            self.__bureau["AMT_CREDIT_SUM_OVERDUE"] / self.__bureau["AMT_ANNUITY"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_ANNUITY"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_ANNUITY"].apply(lambda x: correct(x))
        )

        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"] = (
            # 信用局的逾期总额 /
            # 信用局的当前债务
            self.__bureau["AMT_CREDIT_SUM_OVERDUE"] / self.__bureau["AMT_CREDIT_SUM_DEBT"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"].apply(lambda x: correct(x))
        )

        self.__bureau["NEW_AMT_CREDIT_MAX_OVERDUE_DIVIDE_AMT_CREDIT_SUM_OVERDUE"] = (
            # 最大逾期金额 /
            # 信用局的逾期总额
            self.__bureau["AMT_CREDIT_MAX_OVERDUE"] / self.__bureau["AMT_CREDIT_SUM_OVERDUE"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_MAX_OVERDUE_DIVIDE_AMT_CREDIT_SUM_OVERDUE"] = (
            self.__bureau["NEW_AMT_CREDIT_MAX_OVERDUE_DIVIDE_AMT_CREDIT_SUM_OVERDUE"].apply(lambda x: correct(x))
        )

    def data_return(self):
        # print(self.__bureau.shape)

        return self.__bureau


if __name__ == "__main__":
    pb = PrepareBureau(
        input_path="C:\\Users\\puhui\\Desktop"
    )
    pb.data_prepare()
    pb.data_transform()
    pb.data_generate()
    pb.data_return()