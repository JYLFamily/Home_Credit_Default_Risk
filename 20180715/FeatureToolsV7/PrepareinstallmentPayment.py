# coding:utf-8

import os
import numpy as np
import pandas as pd


class PrepareInstallmentPayment(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__installment_payment = None

        # data transform
        self.__start_time = pd.Timestamp("2018-07-20")

    def data_prepare(self):
        self.__installment_payment = pd.read_csv(os.path.join(self.__input_path, "installments_payments.csv"), nrows=10)
        self.__installment_payment = self.__installment_payment.drop(["SK_ID_CURR"], axis=1)

    def data_transform(self):
        # self.__installment_payment = self.__installment_payment.replace(365243.0, np.nan)
        self.__installment_payment["DAYS_INSTALMENT"] = pd.to_timedelta(self.__installment_payment["DAYS_INSTALMENT"], "D")
        self.__installment_payment["DAYS_ENTRY_PAYMENT"] = pd.to_timedelta(self.__installment_payment["DAYS_ENTRY_PAYMENT"], "D")

        self.__installment_payment["DAYS_INSTALMENT"] += self.__start_time
        self.__installment_payment["DAYS_ENTRY_PAYMENT"] += self.__start_time

        # 方便后续 featuretools 制定 variable types
        for col in self.__installment_payment.columns.tolist():
            if col in self.__installment_payment.select_dtypes(include="object").columns.tolist():
                self.__installment_payment.rename(columns={col: "FLAG_INSTALLMENT_PAYMENT_" + col}, inplace=True)

        self.__installment_payment = pd.get_dummies(
            data=self.__installment_payment,
            prefix="FLAG_INSTALLMENT_PAYMENT",
            dummy_na=True,
            columns=self.__installment_payment.select_dtypes(include="object").columns.tolist()
        )

    def data_generate(self):
        self.__installment_payment["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_BEFORE"] = (
            # 提前还款天数
            # series 需要对每个 time 类型引用 int
            (self.__installment_payment["DAYS_INSTALMENT"] - self.__installment_payment["DAYS_ENTRY_PAYMENT"]).apply(lambda x: x.days)
        ).apply(lambda x: x if x > 0 else 0)

        self.__installment_payment["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_OVERDUE"] = (
            # 逾期天数
            (self.__installment_payment["DAYS_ENTRY_PAYMENT"] - self.__installment_payment["DAYS_INSTALMENT"]).apply(lambda x: x.days)
        ).apply(lambda x: x if x > 0 else 0)

        self.__installment_payment["NEW_AMT_PAYMENT_DIVIDE_AMT_INSTALMENT"] = (
            # 实付款 /
            # 应付款
            self.__installment_payment["AMT_PAYMENT"] /
            self.__installment_payment["AMT_INSTALMENT"].replace(0, np.nan)
        )
        self.__installment_payment["NEW_AMT_PAYMENT_MINUS_AMT_INSTALMENT"] = (
            # 实付款 -
            # 应付款
            self.__installment_payment["AMT_PAYMENT"] - self.__installment_payment["AMT_INSTALMENT"]
        )

    def data_return(self):
        # print(self.__installment_payment.shape)

        return self.__installment_payment


if __name__ == "__main__":
    pip = PrepareInstallmentPayment(
        input_path="C:\\Users\\puhui\\Desktop"
    )
    pip.data_prepare()
    pip.data_transform()
    pip.data_generate()
    pip.data_return()