# coding:utf-8

import re
import numpy as np
import pandas as pd


class ManualFeatureApplication(object):
    def __init__(self, *, application_train, application_test):
        self.__application_train = application_train.copy()
        self.__application_test = application_test.copy()
        self.__income_by_occupation = None

    def add_manual_feature(self):
        self.__income_by_occupation = (
            self.__application_train[["AMT_INCOME_TOTAL", "ORGANIZATION_TYPE"]].groupby("ORGANIZATION_TYPE")["AMT_INCOME_TOTAL"].median()
        )

        for df in [self.__application_train, self.__application_test]:
            # AMT_CREDIT AMT_CREDIT AMT_GOODS_PRICE AMT_INCOME_TOTAL
            df["NEW_CREDIT_TO_ANNUITY_RATIO"] = (
                # Credit amount of the loan / Loan annuity
                df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
            )
            df["NEW_CREDIT_TO_GOODS_RATIO"] = (
                # Credit amount of the loan /
                # For consumer loans it is the price of the goods for which the loan is given
                # 贷款总额 / 贷款购买商品的价格
                df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
            )
            df["NEW_CREDIT_TO_INCOME_RATIO"] = (
                df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
            )
            df["NEW_ANNUITY_TO_INCOME_RATIO"] = (
                df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
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
                    ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
                     "FLAG_EMAIL"]].sum(axis=1)
            )
            df["NEW_CONTACT_IND_KURT"] = (
                df[
                    ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
                     "FLAG_EMAIL"]].kurtosis(axis=1)
            )

            # CNT_CHILDREN
            df["NEW_INC_PER_CHLD"] = (
                # 分母 + 1 防止分母为 0
                df["AMT_INCOME_TOTAL"] / df["CNT_CHILDREN"].replace(0, np.nan)
            )

            # ORGANIZATION_TYPE
            df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].fillna("missing")
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
                df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(0, np.nan)
            )
            df["NEW_CAR_TO_BIRTH_RATIO"] = (
                df["OWN_CAR_AGE"] / df["DAYS_BIRTH"].replace(0, np.nan)
            )
            df["NEW_CAR_TO_EMPLOY_RATIO"] = (
                df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED"].replace(0, np.nan)
            )
            df["NEW_PHONE_TO_BIRTH_RATIO"] = (
                df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"].replace(0, np.nan)
            )
            df["NEW_PHONE_TO_EMPLOY_RATIO"] = (
                df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_EMPLOYED"].replace(0, np.nan)
            )

            # REG_REGION_NOT_LIVE_REGION REG_REGION_NOT_WORK_REGION LIVE_REGION_NOT_WORK_REGION REG_CITY_NOT_LIVE_CITY
            # REG_CITY_NOT_WORK_CITY LIVE_CITY_NOT_WORK_CITY
            df["NEW_REG_IND_SUM"] = (
                df[
                    ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
                     "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].sum(axis=1)
            )

            df["NEW_REG_IND_KURT"] = (
                df[
                    ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
                     "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].kurtosis(axis=1)
            )

        return self.__application_train, self.__application_test


if __name__ == "__main__":
    application_train = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\application_train.csv")
    application_test = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\application_test.csv")

    mfa = ManualFeatureApplication(
        application_train=application_train,
        application_test=application_test
    )
    mfa.add_manual_feature()