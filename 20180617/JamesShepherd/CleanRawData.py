# coding:utf-8

import re
import os
import numpy as np
import pandas as pd


class CleanRawData(object):
    def __init__(self, *, input_path, output_path):
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

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__input_path, "application_train.csv"))
        self.__application_test = pd.read_csv(os.path.join(self.__input_path, "application_test.csv"))
        self.__bureau = pd.read_csv(os.path.join(self.__input_path, "bureau.csv"))
        self.__bureau_balance = pd.read_csv(os.path.join(self.__input_path, "bureau_balance.csv"))
        self.__credit_card_balance = pd.read_csv(os.path.join(self.__input_path, "credit_card_balance.csv"))
        self.__installments_payments = pd.read_csv(os.path.join(self.__input_path, "installments_payments.csv"))
        self.__pos_cash_balance = pd.read_csv(os.path.join(self.__input_path, "POS_CASH_balance.csv"))
        self.__previous_application = pd.read_csv(os.path.join(self.__input_path, "previous_application.csv"))

    def data_clean(self):
        self.__application_train[[i for i in self.__application_train.columns if re.match(r"^DAYS", i)]] = (
            self.__application_train[[i for i in self.__application_train.columns if re.match(r"^DAYS", i)]].replace(
                365243,
                np.nan
            )
        )
        self.__application_train[self.__application_train.select_dtypes("object").columns.tolist()] = (
            self.__application_train[self.__application_train.select_dtypes("object").columns.tolist()].replace(
                ["XNA", "XAP"],
                np.nan
            )
        )

        self.__application_test[[i for i in self.__application_test.columns if re.match(r"^DAYS", i)]] = (
            self.__application_test[[i for i in self.__application_test.columns if re.match(r"^DAYS", i)]].replace(
                365243,
                np.nan
            )
        )
        self.__application_test[self.__application_test.select_dtypes("object").columns.tolist()] = (
            self.__application_test[self.__application_test.select_dtypes("object").columns.tolist()].replace(
                ["XNA", "XAP"],
                np.nan
            )
        )

        # bureau
        self.__bureau[[i for i in self.__bureau.columns if re.match(r"^DAYS", i)]] = (
            self.__bureau[[i for i in self.__bureau.columns if re.match(r"^DAYS", i)]].replace(
                365243,
                np.nan
            )
        )
        self.__bureau[self.__bureau.select_dtypes("object").columns.tolist()] = (
            self.__bureau[self.__bureau.select_dtypes("object").columns.tolist()].replace(
                ["XNA", "XAP"],
                np.nan
            )
        )
        # bureau balance
        self.__bureau_balance[[i for i in self.__bureau_balance.columns if re.match(r"^DAYS", i)]] = (
            self.__bureau_balance[[i for i in self.__bureau_balance.columns if re.match(r"^DAYS", i)]].replace(
                365243,
                np.nan
            )
        )
        self.__bureau_balance[self.__bureau_balance.select_dtypes("object").columns.tolist()] = (
            self.__bureau_balance[self.__bureau_balance.select_dtypes("object").columns.tolist()].replace(
                ["XNA", "XAP"],
                np.nan
            )
        )

        # previous application
        self.__previous_application[[i for i in self.__previous_application.columns if re.match(r"^DAYS", i)]] = (
            self.__previous_application[
                [i for i in self.__previous_application.columns if re.match(r"^DAYS", i)]].replace(
                365243,
                np.nan
            )
        )
        self.__previous_application[self.__previous_application.select_dtypes("object").columns.tolist()] = (
            self.__previous_application[self.__previous_application.select_dtypes("object").columns.tolist()].replace(
                ["XNA", "XAP"],
                np.nan
            )
        )

        # pos cash balance
        self.__pos_cash_balance[[i for i in self.__pos_cash_balance.columns if re.match(r"^DAYS", i)]] = (
            self.__pos_cash_balance[
                [i for i in self.__pos_cash_balance.columns if re.match(r"^DAYS", i)]].replace(
                365243,
                np.nan
            )
        )
        self.__pos_cash_balance[self.__pos_cash_balance.select_dtypes("object").columns.tolist()] = (
            self.__pos_cash_balance[self.__pos_cash_balance.select_dtypes("object").columns.tolist()].replace(
                ["XNA", "XAP"],
                np.nan
            )
        )

        # credit card balance
        self.__credit_card_balance[[i for i in self.__credit_card_balance.columns if re.match(r"^DAYS", i)]] = (
            self.__credit_card_balance[
                [i for i in self.__credit_card_balance.columns if re.match(r"^DAYS", i)]].replace(
                365243,
                np.nan
            )
        )
        self.__credit_card_balance[self.__credit_card_balance.select_dtypes("object").columns.tolist()] = (
            self.__credit_card_balance[self.__credit_card_balance.select_dtypes("object").columns.tolist()].replace(
                ["XNA", "XAP"],
                np.nan
            )
        )

        # installments payments
        self.__installments_payments[[i for i in self.__installments_payments.columns if re.match(r"^DAYS", i)]] = (
            self.__installments_payments[
                [i for i in self.__installments_payments.columns if re.match(r"^DAYS", i)]].replace(
                365243,
                np.nan
            )
        )
        self.__installments_payments[self.__installments_payments.select_dtypes("object").columns.tolist()] = (
            self.__installments_payments[self.__installments_payments.select_dtypes("object").columns.tolist()].replace(
                ["XNA", "XAP"],
                np.nan
            )
        )

    def data_output(self):
        self.__application_train.to_csv(os.path.join(self.__output_path, "application_train.csv"), index=False)
        self.__application_test.to_csv(os.path.join(self.__output_path, "application_test.csv"), index=False)
        self.__bureau.to_csv(os.path.join(self.__output_path, "bureau.csv"), index=False)
        self.__bureau_balance.to_csv(os.path.join(self.__output_path, "bureau_balance.csv"), index=False)
        self.__credit_card_balance.to_csv(os.path.join(self.__output_path, "credit_card_balance.csv"), index=False)
        self.__installments_payments.to_csv(os.path.join(self.__output_path, "installments_payments.csv"), index=False)
        self.__previous_application.to_csv(os.path.join(self.__output_path, "previous_application.csv"), index=False)
        self.__pos_cash_balance.to_csv(os.path.join(self.__output_path, "pos_cash_balance.csv"), index=False)

if __name__ == "__main__":
    crd = CleanRawData(
        input_path="D:\\Kaggle\\Home_Credit_Default_Risk\\raw_data",
        output_path="D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data"
    )
    crd.data_prepare()
    crd.data_clean()
    crd.data_output()

