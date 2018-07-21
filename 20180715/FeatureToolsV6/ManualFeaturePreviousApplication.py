# coding:utf-8

import numpy as np
import pandas as pd


class ManualFeaturePreviousApplication(object):
    def __init__(self, *, previous_application, pos_cash_balance, installments_payments, credit_card_balance):
        self.__previous_application = previous_application.copy()
        self.__pos_cash_balance = pos_cash_balance.copy()
        self.__installments_payments = installments_payments.copy()
        self.__credit_card_balance = credit_card_balance.copy()

    def add_manual_feature(self):
        # clean
        self.__credit_card_balance["AMT_DRAWINGS_ATM_CURRENT"] = [np.nan if i < 0 else i for i in self.__credit_card_balance["AMT_DRAWINGS_ATM_CURRENT"]]
        self.__credit_card_balance["AMT_DRAWINGS_CURRENT"] = [np.nan if i < 0 else i for i in self.__credit_card_balance["AMT_DRAWINGS_CURRENT"]]

        # hand feature
        # 授信额度可能大于申请额度
        self.__previous_application["NEW_AMT_APPLICATION_DIVIDE_AMT_CREDIT"] = (
            # 申请额度 /
            # 授信额度
            self.__previous_application["AMT_APPLICATION"] / self.__previous_application["AMT_CREDIT"].replace(0, np.nan)
        )

        # 预付额度小于授信额度
        self.__previous_application["NEW_AMT_DOWN_PAYMENT_DIVIDE_AMT_CREDIT"] = (
            # 预付额度 /
            # 授信额度
            self.__previous_application["AMT_DOWN_PAYMENT"] / self.__previous_application["AMT_CREDIT"].replace(0, np.nan)
        )
        self.__previous_application["NEW_AMT_DOWN_PAYMENT_DIVIDE_AMT_CREDIT"] = (
            self.__previous_application["NEW_AMT_DOWN_PAYMENT_DIVIDE_AMT_CREDIT"].apply(lambda x: np.clip(x, 0, 1))
        )

        # pos cash balance
        self.__previous_application = self.__previous_application.merge(
            self.__pos_cash_balance[["SK_ID_PREV", "MONTHS_BALANCE"]].groupby("SK_ID_PREV")["MONTHS_BALANCE"].size().to_frame("POS_MONTHS_BALANCE_SIZE"),
            left_on=["SK_ID_PREV"],
            right_index=True,
            how="left"
        )
        self.__pos_cash_balance = self.__pos_cash_balance.drop("MONTHS_BALANCE", axis=1)

        # installments payments
        self.__installments_payments["NEW_AMT_PAYMENT_DIVIDE_AMT_INSTALMENT"] = (
            # 实付款 /
            # 应付款
            self.__installments_payments["AMT_PAYMENT"] / self.__installments_payments["AMT_INSTALMENT"].replace(0, np.nan)
        )
        self.__installments_payments["NEW_AMT_PAYMENT_MINUS_AMT_INSTALMENT"] = (
            # 实付款 -
            # 应付款
            self.__installments_payments["AMT_PAYMENT"] - self.__installments_payments["AMT_INSTALMENT"]
        )

        # 提前天数
        self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_BEFORE"] = (
            # 应还款日期 -
            # 实还款日期
            self.__installments_payments["DAYS_ENTRY_PAYMENT"] - self.__installments_payments["DAYS_INSTALMENT"]
        )
        self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_BEFORE"] = (
            self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_BEFORE"].apply(lambda x: x if x > 0 else 0)
        )

        # 逾期天数
        self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_OVERDUE"] = (
            # 应还款日期 -
            # 实还款日期
            self.__installments_payments["DAYS_ENTRY_PAYMENT"] - self.__installments_payments["DAYS_INSTALMENT"]
        )
        self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_OVERDUE"] = (
            self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_OVERDUE"].apply(lambda x: abs(x) if x < 0 else 0)
        )

        self.__installments_payments = self.__installments_payments.drop(["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"], axis=1)

        # credit card balance
        self.__credit_card_balance["NEW_AMT_BALANCE_DIVIDE_AMT_CREDIT_LIMIT_ACTUAL"] = (
            # 余额可能大于额度
            self.__credit_card_balance["AMT_BALANCE"] / self.__credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
        )

        self.__previous_application = self.__previous_application.merge(
            self.__credit_card_balance[["SK_ID_PREV", "MONTHS_BALANCE"]].groupby("SK_ID_PREV")["MONTHS_BALANCE"].size().to_frame("CREDIT_MONTHS_BALANCE_SIZE"),
            left_on=["SK_ID_PREV"],
            right_index=True,
            how="left"
        )
        self.__credit_card_balance = self.__credit_card_balance.drop("MONTHS_BALANCE", axis=1)

        # 流水表 categorical one hot encoder
        self.__previous_application = pd.get_dummies(
            data=self.__previous_application,
            dummy_na=True,
            columns=self.__previous_application.select_dtypes(include="object").columns.tolist()
        )
        self.__pos_cash_balance = pd.get_dummies(
            data=self.__pos_cash_balance,
            dummy_na=True,
            columns=self.__pos_cash_balance.select_dtypes(include="object").columns.tolist()
        )
        self.__installments_payments = pd.get_dummies(
            data=self.__installments_payments,
            dummy_na=True,
            columns=self.__installments_payments.select_dtypes(include="object").columns.tolist()
        )
        self.__credit_card_balance = pd.get_dummies(
            data=self.__credit_card_balance,
            dummy_na=True,
            columns=self.__credit_card_balance.select_dtypes(include="object").columns.tolist()
        )

        # print(self.__previous_application.columns.tolist())
        # print(self.__pos_cash_balance.columns.tolist())
        # print(self.__installments_payments.head())
        # print(self.__credit_card_balance.head())

        return self.__previous_application, self.__pos_cash_balance, self.__installments_payments, self.__credit_card_balance


if __name__ == "__main__":
    previous_application = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\previous_application.csv")
    pos_cash_balance = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\pos_cash_balance.csv")
    installments_payments = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\installments_payments.csv")
    credit_card_balance = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\credit_card_balance.csv")

    mfpa = ManualFeaturePreviousApplication(
        previous_application=previous_application,
        pos_cash_balance=pos_cash_balance,
        installments_payments=installments_payments,
        credit_card_balance=credit_card_balance
    )
    mfpa.add_manual_feature()