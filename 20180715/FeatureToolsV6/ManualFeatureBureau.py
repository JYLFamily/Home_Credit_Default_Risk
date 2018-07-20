# coding:utf-8

import numpy as np
import pandas as pd


def correct(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1


class ManualFeatureBureau(object):
    def __init__(self, *, bureau, bureau_balance):
        self.__bureau = bureau.copy()
        self.__bureau_balance = bureau_balance.copy()

    def add_manual_feature(self):
        # clean
        self.__bureau["DAYS_CREDIT_ENDDATE"] = [np.nan if i < -4000 else i for i in self.__bureau["DAYS_CREDIT_ENDDATE"]]
        self.__bureau["DAYS_CREDIT_UPDATE"] = [np.nan if i < -4000 else i for i in self.__bureau["DAYS_CREDIT_UPDATE"]]
        self.__bureau["DAYS_ENDDATE_FACT"] = [np.nan if i < -4000 else i for i in self.__bureau["DAYS_ENDDATE_FACT"]]

        # hand feature
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

        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"] = (
            # 信用局的逾期总额 /
            # 信用局的当前债务
            self.__bureau["AMT_CREDIT_SUM_OVERDUE"] / self.__bureau["AMT_CREDIT_SUM_DEBT"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT"].apply(lambda x: correct(x))
        )

        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"] = (
            # 信用局的逾期总额 /
            # 信用局的总额度
            self.__bureau["AMT_CREDIT_SUM_OVERDUE"] / self.__bureau["AMT_CREDIT_SUM"].replace(0, np.nan)
        )
        self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"] = (
            self.__bureau["NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM"].apply(lambda x: correct(x))
        )

        self.__bureau = self.__bureau.merge(
            self.__bureau_balance[["SK_ID_BUREAU", "MONTHS_BALANCE"]].groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].size().to_frame("MONTHS_BALANCE_SIZE"),
            left_on=["SK_ID_BUREAU"],
            right_index=True,
            how="left"
        )
        self.__bureau_balance = self.__bureau_balance.drop("MONTHS_BALANCE", axis=1)

        # 流水表 categorical one hot encoder
        self.__bureau = pd.get_dummies(
            data=self.__bureau,
            dummy_na=True,
            columns=self.__bureau.select_dtypes(include="object").columns.tolist()
        )
        self.__bureau_balance = pd.get_dummies(
            data=self.__bureau_balance,
            dummy_na=True,
            columns=self.__bureau_balance.select_dtypes(include="object").columns.tolist()
        )

        # print(self.__bureau.head(10))
        # print(self.__bureau_balance.head(10))
        # print(np.sum(np.logical_not(self.__bureau["MONTHS_BALANCE_SIZE"].isna())))

        return self.__bureau, self.__bureau_balance

if __name__ == "__main__":
    bureau = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\bureau.csv")
    bureau_balance = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\bureau_balance.csv")

    mfb = ManualFeatureBureau(
        bureau=bureau,
        bureau_balance=bureau_balance
    )
    mfb.add_manual_feature()

