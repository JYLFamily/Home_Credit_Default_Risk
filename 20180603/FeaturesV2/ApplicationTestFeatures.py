# coding:utf-8

import os
import re
import sys
import numpy as np
import pandas as pd
import featuretools as ft


class ApplicationTestFeatures(object):

    def __init__(self, *, input_path, output_path, output_file_name):
        self.__input_path = input_path
        self.__output_path = output_path
        self.__output_file_name = output_file_name

        # data prepare
        self.__application_train = None
        self.__bureau = None
        self.__bureau_balance = None
        self.__previous_application = None
        self.__pos_cash_balance = None
        self.__credit_card_balance = None
        self.__installments_payments = None

        self.__application_train_categorical = None
        self.__bureau_categorical = None
        self.__bureau_balance_categorical = None
        self.__previous_application_categorical = None
        self.__pos_cash_balance_categorical = None
        self.__credit_card_balance_categorical = None
        self.__installments_payments_categorical = None

        # es set
        self.__es = None

        self.__feature_dataframe = None

    def data_prepare(self):
        # application_train
        self.__application_train = pd.read_csv(os.path.join(self.__input_path, "application_test.csv"))
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
        self.__application_train_categorical = dict(zip(
            self.__application_train.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__application_train.select_dtypes("object").columns.tolist()))]
        ))

        # bureau
        self.__bureau = pd.read_csv(os.path.join(self.__input_path, "bureau.csv"))
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
        self.__bureau_categorical = dict(zip(
            self.__bureau.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__bureau.select_dtypes("object").columns.tolist()))]
        ))

        # bureau balance
        self.__bureau_balance = pd.read_csv(os.path.join(self.__input_path, "bureau_balance.csv"))
        self.__bureau_balance["bureau_balance_id"] = list(range(self.__bureau_balance.shape[0]))  # 缺少 index, 添加
        self.__bureau_balance = self.__bureau_balance.drop(["MONTHS_BALANCE"], axis=1)  # MONTHS_BALANCE 与时间相关, 删掉
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
        self.__bureau_balance_categorical = dict(zip(
            self.__bureau_balance.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__bureau_balance.select_dtypes("object").columns.tolist()))]
        ))

        # previous application
        self.__previous_application = pd.read_csv(os.path.join(self.__input_path, "previous_application.csv"))
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
        self.__previous_application_categorical = dict(zip(
            self.__previous_application.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__previous_application.select_dtypes("object").columns.tolist()))]
        ))

        # pos cash balance
        self.__pos_cash_balance = pd.read_csv(os.path.join(self.__input_path, "POS_CASH_balance.csv"))
        self.__pos_cash_balance["pos_cash_balance_id"] = list(range(self.__pos_cash_balance.shape[0]))  # 缺少 index, 添加
        self.__pos_cash_balance = self.__pos_cash_balance.drop(["MONTHS_BALANCE"], axis=1)  # MONTHS_BALANCE 与时间相关, 删掉
        self.__pos_cash_balance = self.__pos_cash_balance.drop(["SK_ID_CURR"], axis=1)
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
        self.__pos_cash_balance_categorical = dict(zip(
            self.__pos_cash_balance.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__pos_cash_balance.select_dtypes("object").columns.tolist()))]
        ))

        # credit card balance
        self.__credit_card_balance = pd.read_csv(os.path.join(self.__input_path, "credit_card_balance.csv"))
        self.__credit_card_balance["credit_card_balance_id"] = list(range(self.__credit_card_balance.shape[0]))  # 缺少 index, 添加
        self.__credit_card_balance = self.__credit_card_balance.drop(["MONTHS_BALANCE"], axis=1)  # MONTHS_BALANCE 与时间相关, 删掉
        self.__credit_card_balance = self.__credit_card_balance.drop(["SK_ID_CURR"], axis=1)
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
        self.__credit_card_balance_categorical = dict(zip(
            self.__credit_card_balance.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__credit_card_balance.select_dtypes("object").columns.tolist()))]
        ))

        # installments payments
        self.__installments_payments = pd.read_csv(os.path.join(self.__input_path, "installments_payments.csv"))
        self.__installments_payments["installments_payments_id"] = list(range(self.__installments_payments.shape[0]))  # 缺少 index, 添加
        self.__installments_payments = self.__installments_payments.drop(["SK_ID_CURR"], axis=1)
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
        self.__installments_payments_categorical = dict(zip(
            self.__installments_payments.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__installments_payments.select_dtypes("object").columns.tolist()))]
        ))

    def es_set(self):
        self.__es = ft.EntitySet(id="application_train")
        self.__es = self.__es.entity_from_dataframe(
            entity_id="application_train",
            dataframe=self.__application_train,
            index="SK_ID_CURR",
            variable_types=None if len(self.__application_train_categorical) == 0 else self.__application_train_categorical
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="bureau",
            dataframe=self.__bureau,
            index="SK_ID_BUREAU",
            variable_types=None if len(self.__bureau_categorical) == 0 else self.__bureau_categorical
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="bureau_balance",
            dataframe=self.__bureau_balance,
            index="bureau_balance_id",
            variable_types=None if len(self.__bureau_balance_categorical) == 0 else self.__bureau_balance_categorical
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="previous_application",
            dataframe=self.__previous_application,
            index="SK_ID_PREV",
            variable_types=None if len(self.__previous_application_categorical) == 0 else self.__previous_application_categorical
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="pos_cash_balance",
            dataframe=self.__pos_cash_balance,
            index="pos_cash_balance_id",
            variable_types=None if len(self.__pos_cash_balance_categorical) == 0 else self.__pos_cash_balance_categorical
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="credit_card_balance",
            dataframe=self.__credit_card_balance,
            index="credit_card_balance_id",
            variable_types=None if len(self.__credit_card_balance_categorical) == 0 else self.__credit_card_balance_categorical
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="installments_payments",
            dataframe=self.__installments_payments,
            index="installments_payments_id",
            variable_types=None if len(self.__installments_payments_categorical) == 0 else self.__installments_payments_categorical
        )

        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["application_train"]["SK_ID_CURR"],
                self.__es["bureau"]["SK_ID_CURR"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["bureau"]["SK_ID_BUREAU"],
                self.__es["bureau_balance"]["SK_ID_BUREAU"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["application_train"]["SK_ID_CURR"],
                self.__es["previous_application"]["SK_ID_CURR"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["previous_application"]["SK_ID_PREV"],
                self.__es["pos_cash_balance"]["SK_ID_PREV"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["previous_application"]["SK_ID_PREV"],
                self.__es["credit_card_balance"]["SK_ID_PREV"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["previous_application"]["SK_ID_PREV"],
                self.__es["installments_payments"]["SK_ID_PREV"]
            )
        )

    def dfs_run(self):
        self.__feature_dataframe, _ = ft.dfs(
            entityset=self.__es,
            target_entity="application_train",
            agg_primitives=[ft.primitives.aggregation_primitives.Sum,
                            ft.primitives.aggregation_primitives.Std,
                            ft.primitives.aggregation_primitives.Max,
                            ft.primitives.aggregation_primitives.Min,
                            ft.primitives.aggregation_primitives.Mean,
                            ft.primitives.aggregation_primitives.Count,
                            ft.primitives.aggregation_primitives.NUnique,
                            ft.primitives.aggregation_primitives.Mode],
            trans_primitives=[],
            verbose=True,
            chunk_size=110  # 调大 chunk_size 以时间换空间, 加大内存占用减少运行时间
        )

        self.__feature_dataframe.to_csv(os.path.join(self.__output_path, self.__output_file_name), index=True)

if __name__ == "__main__":
    atf = ApplicationTestFeatures(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
        output_file_name=sys.argv[3]
    )
    atf.data_prepare()
    atf.es_set()
    atf.dfs_run()
