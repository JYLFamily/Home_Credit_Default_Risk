# coding:utf-8

import os
import re
import importlib
import featuretools as ft
from featuretools.primitives import Sum
from featuretools.primitives import Std
from featuretools.primitives import Max
from featuretools.primitives import Min
from featuretools.primitives import Median
from featuretools.primitives import Count
from featuretools.primitives import PercentTrue
from featuretools.primitives import Trend
from featuretools.primitives import AvgTimeBetween


class FeatureToolsTrainV7(object):

    def __init__(self, input_path, output_path):
        # init
        self.__input_path, self.__output_path = input_path, output_path
        self.__prepare_application_train = importlib.import_module("PrepareApplicationTrain")
        self.__prepare_bureau = importlib.import_module("PrepareBureau")
        self.__prepare_bureau_balance = importlib.import_module("PrepareBureauBalance")
        self.__prepare_previous_application = importlib.import_module("PreparePreviousApplication")
        self.__prepare_credit_card = importlib.import_module("PrepareCreditCard")
        self.__prepare_installment_payment = importlib.import_module("PrepareinstallmentPayment")
        self.__prepare_pos_cash = importlib.import_module("PreparePosCash")

        # data prepare
        self.__application_train_df = None
        self.__bureau_df = None
        self.__bureau_balance_df = None
        self.__previous_application_df = None
        self.__credit_card_df = None
        self.__installment_payment_df = None
        self.__pos_cash_df = None

        # es set
        self.__es = None
        self.__train_feature_matrix = None
        self.__train_feature = None
        self.__application_train_df_variable_types = dict()
        self.__bureau_df_variable_types = dict()
        self.__bureau_balance_df_variable_types = dict()
        self.__previous_application_df_variable_types = dict()
        self.__credit_card_df_variable_types = dict()
        self.__installment_payment_df_variable_types = dict()
        self.__pos_cash_df_variable_types = dict()

    def data_prepare(self):
        pat = self.__prepare_application_train.PrepareApplicationTrain(input_path=self.__input_path)
        pat.data_prepare()
        pat.data_transform()
        pat.data_generate()
        self.__application_train_df = pat.data_return()

        pb = self.__prepare_bureau.PrepareBureau(input_path=self.__input_path)
        pb.data_prepare()
        pb.data_transform()
        pb.data_generate()
        self.__bureau_df = pb.data_return()

        pbb = self.__prepare_bureau_balance.PrepareBureauBalance(input_path=self.__input_path)
        pbb.data_prepare()
        pbb.data_transform()
        pbb.data_generate()
        self.__bureau_balance_df = pbb.data_return()

        ppa = self.__prepare_previous_application.PreparePreviousApplication(input_path=self.__input_path)
        ppa.data_prepare()
        ppa.data_transform()
        ppa.data_generate()
        self.__previous_application_df = ppa.data_return()

        pcc = self.__prepare_credit_card.PrepareCreditCard(input_path=self.__input_path)
        pcc.data_prepare()
        pcc.data_transform()
        pcc.data_generate()
        self.__credit_card_df = pcc.data_return()

        pip = self.__prepare_installment_payment.PrepareInstallmentPayment(input_path=self.__input_path)
        pip.data_prepare()
        pip.data_transform()
        pip.data_generate()
        self.__installment_payment_df = pip.data_return()

        ppc = self.__prepare_pos_cash.PreparePosCash(input_path=self.__input_path)
        ppc.data_prepare()
        ppc.data_transform()
        ppc.data_generate()
        self.__pos_cash_df = ppc.data_return()

    def es_set(self):
        for col in self.__application_train_df.select_dtypes("object").columns.tolist():
            self.__application_train_df_variable_types[col] = ft.variable_types.Categorical

        for col in self.__bureau_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__bureau_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__bureau_balance_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__bureau_balance_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__previous_application_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__previous_application_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__credit_card_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__credit_card_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__installment_payment_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__installment_payment_df_variable_types[col] = ft.variable_types.Boolean

        for col in self.__pos_cash_df.columns.tolist():
            if re.search(r"FLAG", col):
                self.__pos_cash_df_variable_types[col] = ft.variable_types.Boolean

        self.__es = ft.EntitySet(id="application_train")
        self.__es = self.__es.entity_from_dataframe(
            entity_id="application_train",
            dataframe=self.__application_train_df,
            index="SK_ID_CURR",
            variable_types=self.__application_train_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="bureau",
            dataframe=self.__bureau_df,
            index="SK_ID_BUREAU",
            time_index="DAYS_CREDIT",
            variable_types=self.__bureau_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="bureau_balance",
            dataframe=self.__bureau_balance_df,
            make_index=True,
            index="bureau_balance_id",
            time_index="MONTHS_BALANCE",
            variable_types=self.__bureau_balance_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="previous_application",
            dataframe=self.__previous_application_df,
            index="SK_ID_PREV",
            time_index="DAYS_DECISION",
            variable_types=self.__previous_application_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="credit_card",
            dataframe=self.__credit_card_df,
            make_index=True,
            index="credit_card_id",
            time_index="MONTHS_BALANCE",
            variable_types=self.__credit_card_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="installment_payment",
            dataframe=self.__installment_payment_df,
            make_index=True,
            index="installment_payment_id",
            time_index="DAYS_INSTALMENT",
            variable_types=self.__installment_payment_df_variable_types
        )
        self.__es = self.__es.entity_from_dataframe(
            entity_id="pos_cash",
            dataframe=self.__pos_cash_df,
            make_index=True,
            index="pos_cash_id",
            time_index="MONTHS_BALANCE",
            variable_types=self.__pos_cash_df_variable_types
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
                self.__es["pos_cash"]["SK_ID_PREV"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["previous_application"]["SK_ID_PREV"],
                self.__es["credit_card"]["SK_ID_PREV"]
            )
        )
        self.__es = self.__es.add_relationship(
            ft.Relationship(
                self.__es["previous_application"]["SK_ID_PREV"],
                self.__es["installment_payment"]["SK_ID_PREV"]
            )
        )
        self.__es["previous_application"]["FLAG_PREVIOUS_APPLICATION_NAME_CONTRACT_STATUS_Refused"].interesting_values = [1]
        self.__es["previous_application"]["FLAG_PREVIOUS_APPLICATION_NAME_CONTRACT_STATUS_Approved"].interesting_values = [1]
        self.__es["previous_application"]["FLAG_PREVIOUS_APPLICATION_NAME_PRODUCT_TYPE_walk-in"].interesting_values = [1]
        self.__es["previous_application"]["FLAG_PREVIOUS_APPLICATION_CODE_REJECT_REASON_HC"].interesting_values = [1]
        self.__es["bureau"]["FLAG_BUREAU_CREDIT_ACTIVE_Active"].interesting_values = [1]
        self.__es["bureau"]["FLAG_BUREAU_CREDIT_ACTIVE_Closed"].interesting_values = [1]

    def dfs_run(self):
        _, self.__train_feature = ft.dfs(
            entityset=self.__es,
            target_entity="application_train",
            agg_primitives=[Sum, Std, Max, Min, Median, Count, PercentTrue, Trend, AvgTimeBetween],
            where_primitives=[Std, Max, Min, Median, Count],
            verbose=True,
            n_jobs=2,
            chunk_size=10  # 调大 chunk_size 以时间换空间, 加大内存占用减少运行时间
        )

        # ft.save_features(self.__train_feature, os.path.join(self.__output_path, "train_feature_definitions"))
        # self.__train_feature = ft.load_features("train_feature_definitions")

        # self.__train_feature_matrix = ft.calculate_feature_matrix(
        #     features=self.__train_feature,
        #     entityset=self.__es,
        #     n_jobs=2
        # )
        #
        # self.__train_feature_matrix.to_csv(os.path.join(self.__output_path, "train_feature_df.csv"), index=True)

if __name__ == "__main__":
    ftt = FeatureToolsTrainV7(
        input_path="C:\\Users\\puhui\\Desktop",
        output_path="C:\\Users\\puhui\\Desktop"
    )
    ftt.data_prepare()
    ftt.es_set()
    ftt.dfs_run()