# coding:utf-8

import re
import numpy as np


class AddManualFeature(object):
    def __init__(self, *, train_feature, test_feature):
        self.__train_feature = train_feature.copy()
        self.__test_feature = test_feature.copy()
        self.__income_by_occupation = None
        
    def add_manual_feature(self):
        for df in [self.__train_feature, self.__test_feature]:
            # AMT_CREDIT AMT_CREDIT AMT_GOODS_PRICE AMT_INCOME_TOTAL
            df["NEW_CREDIT_TO_ANNUITY_RATIO"] = (
                # Credit amount of the loan / Loan annuity
                df["AMT_CREDIT"] / df["AMT_ANNUITY"]
            )
            df["NEW_CREDIT_TO_GOODS_RATIO"] = (
                # Credit amount of the loan / 
                # For consumer loans it is the price of the goods for which the loan is given
                # 贷款总额 / 贷款购买商品的价格
                df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
            )
            df["NEW_CREDIT_TO_INCOME_RATIO"] = (
                df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
            )
            df["NEW_ANNUITY_TO_INCOME_RATIO"] = (
                df["AMT_ANNUITY"] / (1 + df["AMT_INCOME_TOTAL"])
            )

            # FLAG_DOCUMENT
            # FLAG_OWN_CAR, FLAG_OWN_REALTY
            # FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL
            df["NEW_DOC_IND_SUM"] = (
                df[
                    [col for col in df.columns if re.search(r"FLAG_DOCUMENT", col)]].sum(axis=1)
            )
            df["NEW_DOC_IND_KURT"] = (
                df[
                    [col for col in df.columns if re.search(r"FLAG_DOCUMENT", col)]].kurtosis(axis=1)
            )
            df["NEW_LIVE_IND_SUM"] = (
                df[
                    ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]].sum(axis=1)
            )
            df["NEW_LIVE_IND_KURT"] = (
                df[
                    ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]].kurtosis(axis=1)
            )
            df["NEW_CONTACT_IND_SUM"] = (
                df[
                    ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]].sum(axis=1)
            )
            df["NEW_CONTACT_IND_KURT"] = (
                df[
                    ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]].kurtosis(axis=1)
            )

            # CNT_CHILDREN
            df["NEW_INC_PER_CHLD"] = (
                # 分母 + 1 防止分母为 0
                df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])
            )

            # ORGANIZATION_TYPE
            df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].fillna("missing")
            self.__income_by_occupation = (
                df[["AMT_INCOME_TOTAL", "ORGANIZATION_TYPE"]].groupby("ORGANIZATION_TYPE")["AMT_INCOME_TOTAL"].median()
            )
            df["NEW_INC_BY_ORG"] = df["ORGANIZATION_TYPE"].map(self.__income_by_occupation)

            # EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
            df["NEW_SOURCES_PROD"] = (
                df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
            )
            df["NEW_SOURCES_MEAN"] = (
                df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
            )
            df["NEW_SOURCES_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
            df["NEW_SOURCES_STD"] = df["NEW_SOURCES_STD"].fillna(df["NEW_SOURCES_STD"].mean())
            df["NEW_SOURCES_NA_SUM"] = (
                df["EXT_SOURCE_1"].isna().astype(np.float64) +
                df["EXT_SOURCE_2"].isna().astype(np.float64) +
                df["EXT_SOURCE_3"].isna().astype(np.float64)
            )

            # DAYS_EMPLOYED, DAYS_BIRTH, OWN_CAR_AGE DAYS_LAST_PHONE_CHANGE
            df["NEW_EMPLOY_TO_BIRTH_RATIO"] = (
                # How many days before the application the person started current employment /
                # Client's age in days at the time of application
                # 工龄 / 年龄
                df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
            )
            df["NEW_CAR_TO_BIRTH_RATIO"] = (
                df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]
            )
            df["NEW_CAR_TO_EMPLOY_RATIO"] = (
                df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED"]
            )
            df["NEW_PHONE_TO_BIRTH_RATIO"] = (
                df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
            )
            df["NEW_PHONE_TO_EMPLOY_RATIO"] = (
                df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_EMPLOYED"]
            )

        return self.__train_feature, self.__test_feature