# coding:utf-8

import os
import sys
import pandas as pd


class MergeFeature(object):
    def __init__(self, input_path_v6, input_path_v8, output_path):
        self.__input_path_v6, self.__input_path_v8 = input_path_v6, input_path_v8
        self.__output_path = output_path
        # feature_data_V6 MODE, NUM_UNIQUE feature
        self.__mode_num_unique = [
            "MODE(previous_application.NAME_CONTRACT_TYPE)",
            "MODE(previous_application.WEEKDAY_APPR_PROCESS_START)",
            "MODE(previous_application.FLAG_LAST_APPL_PER_CONTRACT)",
            "MODE(previous_application.NAME_CASH_LOAN_PURPOSE)",
            "MODE(previous_application.NAME_CONTRACT_STATUS)",
            "MODE(previous_application.NAME_PAYMENT_TYPE)",
            "MODE(previous_application.CODE_REJECT_REASON)",
            "MODE(previous_application.NAME_TYPE_SUITE)",
            "MODE(previous_application.NAME_CLIENT_TYPE)",
            "MODE(previous_application.NAME_GOODS_CATEGORY)",
            "MODE(previous_application.NAME_PORTFOLIO)",
            "MODE(previous_application.NAME_PRODUCT_TYPE)",
            "MODE(previous_application.CHANNEL_TYPE)",
            "MODE(previous_application.NAME_SELLER_INDUSTRY)",
            "MODE(previous_application.NAME_YIELD_GROUP)",
            "MODE(previous_application.PRODUCT_COMBINATION)",
            "NUM_UNIQUE(previous_application.NAME_CONTRACT_TYPE)",
            "NUM_UNIQUE(previous_application.WEEKDAY_APPR_PROCESS_START)",
            "NUM_UNIQUE(previous_application.FLAG_LAST_APPL_PER_CONTRACT)",
            "NUM_UNIQUE(previous_application.NAME_CASH_LOAN_PURPOSE)",
            "NUM_UNIQUE(previous_application.NAME_CONTRACT_STATUS)",
            "NUM_UNIQUE(previous_application.NAME_PAYMENT_TYPE)",
            "NUM_UNIQUE(previous_application.CODE_REJECT_REASON)",
            "NUM_UNIQUE(previous_application.NAME_TYPE_SUITE)",
            "NUM_UNIQUE(previous_application.NAME_CLIENT_TYPE)",
            "NUM_UNIQUE(previous_application.NAME_GOODS_CATEGORY)",
            "NUM_UNIQUE(previous_application.NAME_PORTFOLIO)",
            "NUM_UNIQUE(previous_application.NAME_PRODUCT_TYPE)",
            "NUM_UNIQUE(previous_application.CHANNEL_TYPE)",
            "NUM_UNIQUE(previous_application.NAME_SELLER_INDUSTRY)",
            "NUM_UNIQUE(previous_application.NAME_YIELD_GROUP)",
            "NUM_UNIQUE(previous_application.PRODUCT_COMBINATION)",
            "MODE(bureau.CREDIT_ACTIVE)",
            "MODE(bureau.CREDIT_CURRENCY)",
            "MODE(bureau.CREDIT_TYPE)",
            "NUM_UNIQUE(bureau.CREDIT_ACTIVE)",
            "NUM_UNIQUE(bureau.CREDIT_CURRENCY)",
            "NUM_UNIQUE(bureau.CREDIT_TYPE)",
            "MODE(pos_cash_balance.NAME_CONTRACT_STATUS)",
            "NUM_UNIQUE(pos_cash_balance.NAME_CONTRACT_STATUS)",
            "MODE(credit_card_balance.NAME_CONTRACT_STATUS)",
            "NUM_UNIQUE(credit_card_balance.NAME_CONTRACT_STATUS)",
            "MODE(bureau_balance.STATUS)",
            "NUM_UNIQUE(bureau_balance.STATUS)",
            "SUM(previous_application.NUM_UNIQUE(pos_cash_balance.NAME_CONTRACT_STATUS))",
            "SUM(previous_application.NUM_UNIQUE(credit_card_balance.NAME_CONTRACT_STATUS))",
            "STD(previous_application.NUM_UNIQUE(pos_cash_balance.NAME_CONTRACT_STATUS))",
            "STD(previous_application.NUM_UNIQUE(credit_card_balance.NAME_CONTRACT_STATUS))",
            "MAX(previous_application.NUM_UNIQUE(pos_cash_balance.NAME_CONTRACT_STATUS))",
            "MAX(previous_application.NUM_UNIQUE(credit_card_balance.NAME_CONTRACT_STATUS))",
            "MIN(previous_application.NUM_UNIQUE(pos_cash_balance.NAME_CONTRACT_STATUS))",
            "MIN(previous_application.NUM_UNIQUE(credit_card_balance.NAME_CONTRACT_STATUS))",
            "MEDIAN(previous_application.NUM_UNIQUE(pos_cash_balance.NAME_CONTRACT_STATUS))",
            "MEDIAN(previous_application.NUM_UNIQUE(credit_card_balance.NAME_CONTRACT_STATUS))",
            "MODE(previous_application.MODE(pos_cash_balance.NAME_CONTRACT_STATUS))",
            "MODE(previous_application.MODE(credit_card_balance.NAME_CONTRACT_STATUS))",
            "NUM_UNIQUE(previous_application.MODE(pos_cash_balance.NAME_CONTRACT_STATUS))",
            "NUM_UNIQUE(previous_application.MODE(credit_card_balance.NAME_CONTRACT_STATUS))",
            "SUM(bureau.NUM_UNIQUE(bureau_balance.STATUS))",
            "STD(bureau.NUM_UNIQUE(bureau_balance.STATUS))",
            "MAX(bureau.NUM_UNIQUE(bureau_balance.STATUS))",
            "MIN(bureau.NUM_UNIQUE(bureau_balance.STATUS))",
            "MEDIAN(bureau.NUM_UNIQUE(bureau_balance.STATUS))",
            "MODE(bureau.MODE(bureau_balance.STATUS))",
            "NUM_UNIQUE(bureau.MODE(bureau_balance.STATUS))"]
        self.__train_feature_df, self.__test_feature_df = [None for _ in range(2)]
        # feature_data_V8
        self.__train_select_feature_df, self.__test_select_feature_df = [None for _ in range(2)]
        self.__train_final_feature_df, self.__test_final_feature_df = [None for _ in range(2)]

    def data_prepare(self):
        self.__train_feature_df = pd.read_csv(
            os.path.join(self.__input_path_v6, "train_select_feature_df.csv"),
            usecols=self.__mode_num_unique
        )

        self.__test_feature_df = pd.read_csv(
            os.path.join(self.__input_path_v6, "test_select_feature_df.csv"),
            usecols=self.__mode_num_unique
        )

        self.__train_select_feature_df = pd.read_csv(
            os.path.join(self.__input_path_v8, "train_select_feature_df.csv")
        )

        self.__test_select_feature_df = pd.read_csv(
            os.path.join(self.__input_path_v8, "test_select_feature_df.csv")
        )

        self.__train_final_feature_df = pd.concat([self.__train_feature_df, self.__train_select_feature_df], axis=1)
        self.__test_final_feature_df = pd.concat([self.__test_feature_df, self.__test_select_feature_df], axis=1)

    def data_output(self):
        self.__train_final_feature_df.to_csv(
            os.path.join(self.__output_path, "train_select_feature_df.csv"),
            index=False
        )

        self.__test_final_feature_df.to_csv(
            os.path.join(self.__output_path, "test_select_feature_df.csv"),
            index=False
        )


if __name__ == "__main__":
    mf = MergeFeature(
        input_path_v6=sys.argv[1],
        input_path_v8=sys.argv[2],
        output_path=sys.argv[3]
    )
    mf.data_prepare()
    mf.data_output()
