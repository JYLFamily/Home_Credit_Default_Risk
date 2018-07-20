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

        for i, df in enumerate([self.__application_train, self.__application_test]):
            # clean
            df["DAYS_LAST_PHONE_CHANGE"] = (
                df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan)
            )

            # hand feature
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
            df["NEW_SOURCES_MIN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
            df["NEW_SOURCES_MAX"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
            df["NEW_SOURCES_SUM"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].sum(axis=1)

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

            # neptune.ml
            df["NEW_CHILDREN_RATIO"] = (
                df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
            )

            df["NEW_INCOME_CREDIT_PERCENTAGE"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"].replace(0, np.nan)

            df["NEW_INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

            df["NEW_PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)

            df["NEW_CNT_NON_CHILD"] = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]

            df["NEW_CHILD_TO_NON_CHILD_RATIO"] = (
                df["CNT_CHILDREN"] / df["NEW_CNT_NON_CHILD"].replace(0, np.nan)
            )

            df["NEW_INCOME_PER_NON_CHILD"] = df["AMT_INCOME_TOTAL"] / df["NEW_CNT_NON_CHILD"].replace(0, np.nan)

            df["NEW_CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
            df["NEW_CREDIT_PER_CHILD"] = df["AMT_CREDIT"] / df["CNT_CHILDREN"].replace(0, np.nan)
            df["NEW_CREDIT_PER_NON_CHILD"] = df["AMT_CREDIT"] / df["NEW_CNT_NON_CHILD"].replace(0, np.nan)

            df["NEW_RETIREMENT_AGE"] = (df["DAYS_BIRTH"] < -14000).astype(int)
            df["NEW_LONG_EMPLOYMENT"] = (df["DAYS_EMPLOYED"] < -2000).astype(int)

            NEW_AGGREGATION_RECIPIES = [
                (["CODE_GENDER",
                  "NAME_EDUCATION_TYPE"], [("AMT_ANNUITY",  "max"),
                                           ("AMT_CREDIT",   "max"),
                                           ("EXT_SOURCE_1", "mean"),
                                           ("EXT_SOURCE_2", "mean"),
                                           ("OWN_CAR_AGE",  "max"),
                                           ("OWN_CAR_AGE",  "sum"),
                                           ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                           ("NEW_SOURCES_MEAN", "median"),
                                           ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                           ("NEW_SOURCES_PROD", "median"),
                                           ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                           ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                           ("NEW_SOURCES_STD", "median"),
                                           ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                           ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                           ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

                (["CODE_GENDER",
                  "ORGANIZATION_TYPE"],   [("AMT_ANNUITY",       "mean"),
                                           ("AMT_INCOME_TOTAL",  "mean"),
                                           ("DAYS_REGISTRATION", "mean"),
                                           ("EXT_SOURCE_1",      "mean"),
                                           ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                           ("NEW_SOURCES_MEAN", "median"),
                                           ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                           ("NEW_SOURCES_PROD", "median"),
                                           ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                           ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                           ("NEW_SOURCES_STD", "median"),
                                           ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                           ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                           ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

                (["CODE_GENDER",
                  "REG_CITY_NOT_WORK_CITY"], [("AMT_ANNUITY",      "mean"),
                                              ("CNT_CHILDREN",    "mean"),
                                              ("DAYS_ID_PUBLISH", "mean"),
                                              ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                              ("NEW_SOURCES_MEAN", "median"),
                                              ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                              ("NEW_SOURCES_PROD", "median"),
                                              ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                              ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                              ("NEW_SOURCES_STD", "median"),
                                              ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                              ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                              ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

                (["CODE_GENDER",
                  "NAME_EDUCATION_TYPE",
                  "OCCUPATION_TYPE",
                  "REG_CITY_NOT_WORK_CITY"], [("EXT_SOURCE_1", "mean"),
                                              ("EXT_SOURCE_2", "mean"),
                                              ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                              ("NEW_SOURCES_MEAN", "median"),
                                              ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                              ("NEW_SOURCES_PROD", "median"),
                                              ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                              ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                              ("NEW_SOURCES_STD", "median"),
                                              ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                              ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                              ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),
                (["NAME_EDUCATION_TYPE",
                  "OCCUPATION_TYPE"],        [("AMT_CREDIT",                 "mean"),
                                              ("AMT_REQ_CREDIT_BUREAU_YEAR", "mean"),
                                              ("APARTMENTS_AVG",             "mean"),
                                              ("BASEMENTAREA_AVG",           "mean"),
                                              ("EXT_SOURCE_1",               "mean"),
                                              ("EXT_SOURCE_2",               "mean"),
                                              ("EXT_SOURCE_3",               "mean"),
                                              ("NONLIVINGAREA_AVG",          "mean"),
                                              ("OWN_CAR_AGE",                "mean"),
                                              ("YEARS_BUILD_AVG",            "mean"),
                                              ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                              ("NEW_SOURCES_MEAN", "median"),
                                              ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                              ("NEW_SOURCES_PROD", "median"),
                                              ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                              ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                              ("NEW_SOURCES_STD", "median"),
                                              ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                              ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                              ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

                (["NAME_EDUCATION_TYPE",
                  "OCCUPATION_TYPE",
                  "REG_CITY_NOT_WORK_CITY"], [("ELEVATORS_AVG", "mean"),
                                              ("EXT_SOURCE_1",  "mean"),
                                              ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                              ("NEW_SOURCES_MEAN", "median"),
                                              ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                              ("NEW_SOURCES_PROD", "median"),
                                              ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                              ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                              ("NEW_SOURCES_STD", "median"),
                                              ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                              ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                              ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

                (["OCCUPATION_TYPE"],        [("AMT_ANNUITY",     "mean"),
                                              ("CNT_CHILDREN",    "mean"),
                                              ("CNT_FAM_MEMBERS", "mean"),
                                              ("DAYS_BIRTH",      "mean"),
                                              ("DAYS_EMPLOYED",   "mean"),
                                              ("DAYS_ID_PUBLISH", "mean"),
                                              ("DAYS_REGISTRATION", "mean"),
                                              ("EXT_SOURCE_1", "mean"),
                                              ("EXT_SOURCE_2", "mean"),
                                              ("EXT_SOURCE_3", "mean"),
                                              ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                              ("NEW_SOURCES_MEAN", "median"),
                                              ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                              ("NEW_SOURCES_PROD", "median"),
                                              ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                              ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                              ("NEW_SOURCES_STD", "median"),
                                              ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                              ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                              ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),
            ]

            # 这里的 copy() 跟上面的不同
            for groupby_cols, specs in NEW_AGGREGATION_RECIPIES:
                group_object = df.groupby(groupby_cols)
                for select, agg in specs:
                    groupby_aggregate_name = "{}_{}_{}_{}".format("NEW", "_".join(groupby_cols), agg, select)
                    df = df.merge(
                        group_object[select]
                            .agg(agg)
                            .reset_index()
                            .rename(index=str, columns={select: groupby_aggregate_name}),
                        left_on=groupby_cols,
                        right_on=groupby_cols,
                        how="left"
                    )

            if i == 0:
                self.__application_train = df
            else:
                self.__application_test = df

        return self.__application_train, self.__application_test


if __name__ == "__main__":
    application_train = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\application_train.csv", nrows=100)
    application_test = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\application_test.csv", nrows=100)

    mfa = ManualFeatureApplication(
        application_train=application_train,
        application_test=application_test
    )
    mfa.add_manual_feature()