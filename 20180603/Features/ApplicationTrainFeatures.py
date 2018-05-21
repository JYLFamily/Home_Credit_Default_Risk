# coding:utf-8

import os
import pandas as pd
import featuretools as ft


class ApplicationTrainFeatures(object):

    def __init__(self, *, path):
        self.__path = path
        self.__application_train = None
        self.__bureau = None
        self.__bureau_balance = None
        self.__previous_application = None
        self.__pos_cash_balance = None
        self.__credit_card_balance = None
        self.__installments_payments = None

        self.__es = None

        self.__feature_dataframe = None

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__path, "application_train.csv"))
        self.__bureau = pd.read_csv(os.path.join(self.__path, "bureau.csv"))

        self.__bureau_balance = pd.read_csv(os.path.join(self.__path, "bureau_balance.csv"))
        self.__bureau_balance["bureau_balance_id"] = list(range(self.__bureau_balance.shape[0]))  # 缺少 index, 添加
        self.__bureau_balance = self.__bureau_balance.drop(["MONTHS_BALANCE"], axis=1)  # MONTHS_BALANCE 与时间相关, 删掉

        self.__previous_application = pd.read_csv(os.path.join(self.__path, "previous_application.csv"))

        self.__pos_cash_balance = pd.read_csv(os.path.join(self.__path, "POS_CASH_balance.csv"))
        self.__pos_cash_balance["pos_cash_balance_id"] = list(range(self.__pos_cash_balance.shape[0]))  # 缺少 index, 添加
        self.__pos_cash_balance = self.__pos_cash_balance.drop(["MONTHS_BALANCE"], axis=1)  # MONTHS_BALANCE 与时间相关, 删掉
        self.__pos_cash_balance = self.__pos_cash_balance.drop(["SK_ID_CURR"], axis=1)

        self.__credit_card_balance = pd.read_csv(os.path.join(self.__path, "credit_card_balance.csv"))
        self.__credit_card_balance["credit_card_balance_id"] = list(range(self.__credit_card_balance.shape[0]))  # 缺少 index, 添加
        self.__credit_card_balance = self.__credit_card_balance.drop(["MONTHS_BALANCE"], axis=1)  # MONTHS_BALANCE 与时间相关, 删掉
        self.__credit_card_balance = self.__credit_card_balance.drop(["SK_ID_CURR"], axis=1)

        self.__installments_payments = pd.read_csv(os.path.join(self.__path, "installments_payments.csv"))
        self.__installments_payments["installments_payments_id"] = list(range(self.__installments_payments.shape[0]))  # 缺少 index, 添加
        self.__installments_payments = self.__installments_payments.drop(["SK_ID_CURR"], axis=1)

    def es_set(self):
        self.__es = ft.EntitySet(id="application_train")
        self.__es = self.__es.entity_from_dataframe(
            entity_id="application_train",
            dataframe=self.__application_train,
            index="SK_ID_CURR"
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="bureau",
            dataframe=self.__bureau,
            index="SK_ID_BUREAU",
            variable_types={
                "CREDIT_ACTIVE": ft.variable_types.Categorical,
                "CREDIT_CURRENCY": ft.variable_types.Categorical,
                "CREDIT_TYPE": ft.variable_types.Categorical
            }
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="bureau_balance",
            dataframe=self.__bureau_balance,
            index="bureau_balance_id",
            variable_types={
                "STATUS": ft.variable_types.Categorical
            }
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="previous_application",
            dataframe=self.__previous_application,
            index="SK_ID_PREV",
            variable_types={
                "NAME_CONTRACT_TYPE": ft.variable_types.Categorical,
                "WEEKDAY_APPR_PROCESS_START": ft.variable_types.Categorical,
                "FLAG_LAST_APPL_PER_CONTRACT": ft.variable_types.Categorical,
                "NAME_CASH_LOAN_PURPOSE": ft.variable_types.Categorical,
                "NAME_CONTRACT_STATUS": ft.variable_types.Categorical,
                "NAME_PAYMENT_TYPE": ft.variable_types.Categorical,
                "CODE_REJECT_REASON": ft.variable_types.Categorical,
                "NAME_TYPE_SUITE": ft.variable_types.Categorical,
                "NAME_CLIENT_TYPE": ft.variable_types.Categorical,
                "NAME_GOODS_CATEGORY": ft.variable_types.Categorical,
                "NAME_PORTFOLIO": ft.variable_types.Categorical,
                "NAME_PRODUCT_TYPE": ft.variable_types.Categorical,
                "CHANNEL_TYPE": ft.variable_types.Categorical,
                "NAME_SELLER_INDUSTRY": ft.variable_types.Categorical,
                "NAME_YIELD_GROUP": ft.variable_types.Categorical,
                "PRODUCT_COMBINATION": ft.variable_types.Categorical
            }
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="pos_cash_balance",
            dataframe=self.__pos_cash_balance,
            index="pos_cash_balance_id",
            variable_types={
                "NAME_CONTRACT_STATUS": ft.variable_types.Categorical
            }
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="credit_card_balance",
            dataframe=self.__credit_card_balance,
            index="credit_card_balance_id"
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="installments_payments",
            dataframe=self.__installments_payments,
            index="installments_payments_id"
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
            verbose=True
        )

if __name__ == "__main__":
    atf = ApplicationTrainFeatures(
        path="D:\\Kaggle\\Home_Credit_Default_Risk\\raw_data\\"
    )
    atf.data_prepare()
    atf.es_set()
    atf.dfs_run()
