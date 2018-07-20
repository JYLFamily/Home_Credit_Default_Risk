# coding:utf-8

import os
import sys
import importlib
import pandas as pd
import featuretools as ft
from featuretools.primitives.aggregation_primitives import Sum
from featuretools.primitives.aggregation_primitives import Std
from featuretools.primitives.aggregation_primitives import Min
from featuretools.primitives.aggregation_primitives import Max
from featuretools.primitives.aggregation_primitives import Median
from featuretools.primitives.aggregation_primitives import Count
from featuretools.primitives.aggregation_primitives import Skew


class ApplicationTestFeatureToolsV6(object):
    def __init__(self, *, input_path, output_path, output_file_name):
        self.__ManualFeatureApplicationPy = importlib.import_module("ManualFeatureApplication")
        self.__ManualFeatureBureauPy = importlib.import_module("ManualFeatureBureau")
        self.__ManualFeaturePreviousApplicationPy = importlib.import_module("ManualFeaturePreviousApplication")
        self.__input_path = input_path
        self.__output_path = output_path
        self.__output_file_name = output_file_name

        # data prepare
        self.__application_train = None
        self.__application_test = None
        self.__bureau = None
        self.__bureau_balance = None
        self.__previous_application = None
        self.__pos_cash_balance = None
        self.__credit_card_balance = None
        self.__installments_payments = None

        self.__application_test_categorical = None
        self.__bureau_categorical = None
        self.__bureau_balance_categorical = None
        self.__previous_application_categorical = None
        self.__pos_cash_balance_categorical = None
        self.__credit_card_balance_categorical = None
        self.__installments_payments_categorical = None

        # es set
        self.__es = None
        self.__feature = None

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__input_path, "application_train.csv"))
        self.__application_test = pd.read_csv(os.path.join(self.__input_path, "application_test.csv"))
        self.__bureau = pd.read_csv(os.path.join(self.__input_path, "bureau.csv"))
        self.__bureau_balance = pd.read_csv(os.path.join(self.__input_path, "bureau_balance.csv"))
        self.__credit_card_balance = pd.read_csv(os.path.join(self.__input_path, "credit_card_balance.csv"))
        self.__installments_payments = pd.read_csv(os.path.join(self.__input_path, "installments_payments.csv"))
        self.__pos_cash_balance = pd.read_csv(os.path.join(self.__input_path, "POS_CASH_balance.csv"))
        self.__previous_application = pd.read_csv(os.path.join(self.__input_path, "previous_application.csv"))

        _, self.__application_test = (
            self.__ManualFeatureApplicationPy.ManualFeatureApplication(
                application_train=self.__application_train,
                application_test=self.__application_test
            ).add_manual_feature()
        )
        self.__bureau, self.__bureau_balance = (
            self.__ManualFeatureBureauPy.ManualFeatureBureau(
                bureau=self.__bureau,
                bureau_balance=self.__bureau_balance
            ).add_manual_feature()
        )
        self.__previous_application, self.__pos_cash_balance, self.__installments_payments, self.__credit_card_balance = (
            self.__ManualFeaturePreviousApplicationPy.ManualFeaturePreviousApplication(
                previous_application=self.__previous_application,
                pos_cash_balance=self.__pos_cash_balance,
                installments_payments=self.__installments_payments,
                credit_card_balance=self.__credit_card_balance
            ).add_manual_feature()
        )

        self.__application_test_categorical = dict(zip(
            self.__application_test.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__application_test.select_dtypes("object").columns.tolist()))]
        ))
        self.__bureau_categorical = dict(zip(
            self.__bureau.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__bureau.select_dtypes("object").columns.tolist()))]
        ))
        self.__bureau_balance_categorical = dict(zip(
            self.__bureau_balance.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__bureau_balance.select_dtypes("object").columns.tolist()))]
        ))
        self.__previous_application_categorical = dict(zip(
            self.__previous_application.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__previous_application.select_dtypes("object").columns.tolist()))]
        ))
        self.__pos_cash_balance_categorical = dict(zip(
            self.__pos_cash_balance.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__pos_cash_balance.select_dtypes("object").columns.tolist()))]
        ))
        self.__credit_card_balance_categorical = dict(zip(
            self.__credit_card_balance.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__credit_card_balance.select_dtypes("object").columns.tolist()))]
        ))
        self.__installments_payments_categorical = dict(zip(
            self.__installments_payments.select_dtypes("object").columns.tolist(),
            [ft.variable_types.Categorical for _ in range(len(self.__installments_payments.select_dtypes("object").columns.tolist()))]
        ))

    def es_set(self):
        self.__es = ft.EntitySet(id="application_test")
        self.__es = self.__es.entity_from_dataframe(
            entity_id="application_test",
            dataframe=self.__application_test,
            index="SK_ID_CURR",
            variable_types=None if len(self.__application_test_categorical) == 0 else self.__application_test_categorical
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
            make_index=True,
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
            make_index=True,
            index="pos_cash_balance_id",
            variable_types=None if len(self.__pos_cash_balance_categorical) == 0 else self.__pos_cash_balance_categorical
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="credit_card_balance",
            dataframe=self.__credit_card_balance,
            make_index=True,
            index="credit_card_balance_id",
            variable_types=None if len(self.__credit_card_balance_categorical) == 0 else self.__credit_card_balance_categorical
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="installments_payments",
            dataframe=self.__installments_payments,
            make_index=True,
            index="installments_payments_id",
            variable_types=None if len(self.__installments_payments_categorical) == 0 else self.__installments_payments_categorical
        )

        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["application_test"]["SK_ID_CURR"],
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
                self.__es["application_test"]["SK_ID_CURR"],
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
        self.__es["previous_application"]["NAME_CONTRACT_STATUS_Refused"].interesting_values = [1]
        self.__es["previous_application"]["NAME_CONTRACT_STATUS_Approved"].interesting_values = [1]
        self.__es["previous_application"]["NAME_PRODUCT_TYPE_walk-in"].interesting_values = [1]
        self.__es["previous_application"]["CODE_REJECT_REASON_HC"].interesting_values = [1]
        self.__es["bureau"]["CREDIT_ACTIVE_Active"].interesting_values = [1]
        self.__es["bureau"]["CREDIT_ACTIVE_Closed"].interesting_values = [1]

    def dfs_run(self):
        self.__feature, _ = ft.dfs(
            entityset=self.__es,
            target_entity="application_test",
            agg_primitives=[Sum, Std, Max, Min, Median, Count, Skew],
            trans_primitives=[],
            verbose=True,
            chunk_size=110  # 调大 chunk_size 以时间换空间, 加大内存占用减少运行时间
        )

        self.__feature.to_csv(os.path.join(self.__output_path, self.__output_file_name), index=True)


if __name__ == "__main__":
    atftv4 = ApplicationTestFeatureToolsV6(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
        output_file_name="test_feature_df.csv"
    )
    atftv4.data_prepare()
    atftv4.es_set()
    atftv4.dfs_run()
