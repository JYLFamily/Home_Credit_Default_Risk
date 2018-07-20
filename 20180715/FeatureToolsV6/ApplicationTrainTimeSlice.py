# coding:utf-8

import os
import sys
import tqdm
import numpy as np
import pandas as pd


class ApplicationTrainTimeSlice(object):
    def __init__(self, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # data prepare
        self.__application_train = None
        self.__pos_cash_balance = None
        self.__installments_payments = None
        self.__credit_card_balance = None

    def data_prepare(self):
        self.__application_train = pd.read_csv(
            os.path.join(self.__input_path, "application_train.csv"),
            usecols=["SK_ID_CURR"]
        )
        self.__pos_cash_balance = pd.read_csv(os.path.join(self.__input_path, "POS_CASH_balance.csv"))
        self.__installments_payments = pd.read_csv(os.path.join(self.__input_path, "installments_payments.csv"))
        self.__credit_card_balance = pd.read_csv(os.path.join(self.__input_path, "credit_card_balance.csv"))

    def calc_pos_time_slice_features(self):
        self.__pos_cash_balance = self.__pos_cash_balance[[col for col in self.__pos_cash_balance.columns if col != "SK_ID_PREV"]]
        self.__pos_cash_balance = pd.get_dummies(
            data=self.__pos_cash_balance,
            dummy_na=True,
            columns=self.__pos_cash_balance.select_dtypes(include="object").columns.tolist()
        )
        self.__pos_cash_balance["MONTHS_BALANCE"] = self.__pos_cash_balance["MONTHS_BALANCE"].abs()

        for i in tqdm.tqdm(list(np.percentile(self.__pos_cash_balance["MONTHS_BALANCE"], [25, 50, 75]))):
            # filter
            i = int(i)
            temp = self.__pos_cash_balance.loc[
                self.__pos_cash_balance["MONTHS_BALANCE"] <= i,
                [col for col in self.__pos_cash_balance.columns if col != "MONTHS_BALANCE"]
            ]

            # min
            temp_min = temp.groupby(["SK_ID_CURR"]).min().reset_index()
            temp_min.columns = (
                ["SK_ID_CURR"] + [col + str("_min") + str("_") + str(i) for col in temp_min.columns if col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_min,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # max
            temp_max = temp.groupby(["SK_ID_CURR"]).max().reset_index()
            temp_max.columns = (
                ["SK_ID_CURR"] + [col + str("_max") + str("_") + str(i) for col in temp_max.columns if col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_max,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # std
            temp_std = temp.groupby(["SK_ID_CURR"]).std().reset_index()
            temp_std.columns = (
                ["SK_ID_CURR"] + [col + str("_std") + str("_") + str(i) for col in temp_std.columns if col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_std,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # count
            temp_count = temp.groupby(["SK_ID_CURR"]).count().reset_index()
            temp_count.columns = (
                ["SK_ID_CURR"] + [col + str("_count") + str("_") + str(i) for col in temp_count.columns if col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_count,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # median
            temp_median = temp.groupby(["SK_ID_CURR"]).median().reset_index()
            temp_median.columns = (
                ["SK_ID_CURR"] + [col + str("_median") + str("_") + str(i) for col in temp_median.columns if col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_median,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

    def calc_credit_card_time_slice_features(self):
        self.__credit_card_balance = self.__credit_card_balance[
            [col for col in self.__credit_card_balance.columns if col != "SK_ID_PREV"]]
        self.__credit_card_balance = pd.get_dummies(
            data=self.__credit_card_balance,
            dummy_na=True,
            columns=self.__credit_card_balance.select_dtypes(include="object").columns.tolist()
        )
        self.__credit_card_balance["MONTHS_BALANCE"] = self.__credit_card_balance["MONTHS_BALANCE"].abs()

        for i in tqdm.tqdm(list(np.percentile(self.__credit_card_balance["MONTHS_BALANCE"], [25, 50, 75]))):
            # filter
            i = int(i)
            temp = self.__credit_card_balance.loc[
                self.__credit_card_balance["MONTHS_BALANCE"] <= i,
                [col for col in self.__credit_card_balance.columns if col != "MONTHS_BALANCE"]
            ]

            # min
            temp_min = temp.groupby(["SK_ID_CURR"]).min().reset_index()
            temp_min.columns = (
                ["SK_ID_CURR"] + [col + str("_min") + str("_") + str(i) for col in temp_min.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_min,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # max
            temp_max = temp.groupby(["SK_ID_CURR"]).max().reset_index()
            temp_max.columns = (
                ["SK_ID_CURR"] + [col + str("_max") + str("_") + str(i) for col in temp_max.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_max,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # std
            temp_std = temp.groupby(["SK_ID_CURR"]).std().reset_index()
            temp_std.columns = (
                ["SK_ID_CURR"] + [col + str("_std") + str("_") + str(i) for col in temp_std.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_std,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # count
            temp_count = temp.groupby(["SK_ID_CURR"]).count().reset_index()
            temp_count.columns = (
                ["SK_ID_CURR"] + [col + str("_count") + str("_") + str(i) for col in temp_count.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_count,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # median
            temp_median = temp.groupby(["SK_ID_CURR"]).median().reset_index()
            temp_median.columns = (
                ["SK_ID_CURR"] + [col + str("_median") + str("_") + str(i) for col in temp_median.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_median,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

    def calc_installments_payments_time_slice_features(self):
        self.__installments_payments["NEW_AMT_PAYMENT_DIVIDE_AMT_INSTALMENT"] = (
            # 实付款 /
            # 应付款
            self.__installments_payments["AMT_PAYMENT"] / self.__installments_payments["AMT_INSTALMENT"].replace(0,
                                                                                                                 np.nan)
        )
        self.__installments_payments["NEW_AMT_PAYMENT_MINUS_AMT_INSTALMENT"] = (
            # 实付款 /
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
            self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_BEFORE"].apply(
                lambda x: x if x > 0 else 0)
        )

        # 逾期天数
        self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_OVERDUE"] = (
            # 应还款日期 -
            # 实还款日期
            self.__installments_payments["DAYS_ENTRY_PAYMENT"] - self.__installments_payments["DAYS_INSTALMENT"]
        )
        self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_OVERDUE"] = (
            self.__installments_payments["NEW_DAYS_ENTRY_PAYMENT_MINUS_DAYS_INSTALMENT_OVERDUE"].apply(
                lambda x: abs(x) if x < 0 else 0)
        )

        self.__installments_payments = self.__installments_payments.drop(["DAYS_ENTRY_PAYMENT"], axis=1)

        self.__installments_payments = self.__installments_payments[
            [col for col in self.__installments_payments.columns if col != "SK_ID_PREV"]]
        self.__installments_payments = pd.get_dummies(
            data=self.__installments_payments,
            dummy_na=True,
            columns=self.__installments_payments.select_dtypes(include="object").columns.tolist()
        )
        self.__installments_payments["DAYS_INSTALMENT"] = self.__installments_payments["DAYS_INSTALMENT"].abs()

        for i in tqdm.tqdm(list(np.percentile(self.__installments_payments["DAYS_INSTALMENT"], [25, 50, 75]))):
            # filter
            i = int(i)
            temp = self.__installments_payments.loc[
                self.__installments_payments["DAYS_INSTALMENT"] <= i,
                [col for col in self.__installments_payments.columns if col != "DAYS_INSTALMENT"]
            ]

            # min
            temp_min = temp.groupby(["SK_ID_CURR"]).min().reset_index()
            temp_min.columns = (
                ["SK_ID_CURR"] + [col + str("_min") + str("_") + str(i) for col in temp_min.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_min,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # max
            temp_max = temp.groupby(["SK_ID_CURR"]).max().reset_index()
            temp_max.columns = (
                ["SK_ID_CURR"] + [col + str("_max") + str("_") + str(i) for col in temp_max.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_max,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # std
            temp_std = temp.groupby(["SK_ID_CURR"]).std().reset_index()
            temp_std.columns = (
                ["SK_ID_CURR"] + [col + str("_std") + str("_") + str(i) for col in temp_std.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_std,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # count
            temp_count = temp.groupby(["SK_ID_CURR"]).count().reset_index()
            temp_count.columns = (
                ["SK_ID_CURR"] + [col + str("_count") + str("_") + str(i) for col in temp_count.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_count,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

            # median
            temp_median = temp.groupby(["SK_ID_CURR"]).median().reset_index()
            temp_median.columns = (
                ["SK_ID_CURR"] + [col + str("_median") + str("_") + str(i) for col in temp_median.columns if
                                  col != "SK_ID_CURR"]
            )
            self.__application_train = self.__application_train.merge(
                temp_median,
                left_on=["SK_ID_CURR"],
                right_on=["SK_ID_CURR"],
                how="left"
            )

    def data_output(self):
        self.__application_train.to_csv(
            os.path.join(self.__output_path, "application_train_time_slice.csv"),
            index=False
        )


if __name__ == "__main__":
    atts = ApplicationTrainTimeSlice(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    atts.data_prepare()
    atts.calc_pos_time_slice_features()
    atts.calc_credit_card_time_slice_features()
    atts.calc_installments_payments_time_slice_features()
    atts.data_output()



