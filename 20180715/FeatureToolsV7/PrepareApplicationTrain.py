# coding:utf-8

import os
import re
import numpy as np
import pandas as pd


class PrepareApplicationTrain(object):

    def __init__(self, *, input_path):
        self.__input_path = input_path

        # data prepare
        self.__application_train = None

    def data_prepare(self):
        self.__application_train = pd.read_csv(os.path.join(self.__input_path, "application_train.csv"), nrows=10)
        
    def data_transform(self):
        pass
    
    def data_generate(self):
        # AMT_CREDIT AMT_CREDIT AMT_GOODS_PRICE AMT_INCOME_TOTAL
        self.__application_train["NEW_CREDIT_TO_ANNUITY_RATIO"] = (
            # Credit amount of the loan / Loan annuity
            self.__application_train["AMT_CREDIT"] / self.__application_train["AMT_ANNUITY"].replace(0, np.nan)
        )
        self.__application_train["NEW_CREDIT_TO_GOODS_RATIO"] = (
            # Credit amount of the loan /
            # For consumer loans it is the price of the goods for which the loan is given
            # 贷款总额 / 贷款购买商品的价格
            self.__application_train["AMT_CREDIT"] / self.__application_train["AMT_GOODS_PRICE"].replace(0, np.nan)
        )
        self.__application_train["NEW_CREDIT_TO_INCOME_RATIO"] = (
            self.__application_train["AMT_CREDIT"] / self.__application_train["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )
        self.__application_train["NEW_ANNUITY_TO_INCOME_RATIO"] = (
            self.__application_train["AMT_ANNUITY"] / self.__application_train["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )

        # FLAG_DOCUMENT
        # FLAG_OWN_CAR, FLAG_OWN_REALTY
        # FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL
        self.__application_train["NEW_DOC_IND_SUM"] = (
            self.__application_train[
                [col for col in self.__application_train.columns if re.search(r"FLAG_DOCUMENT", col)]].sum(axis=1)
        )
        self.__application_train["NEW_DOC_IND_KURT"] = (
            self.__application_train[
                [col for col in self.__application_train.columns if re.search(r"FLAG_DOCUMENT", col)]].kurtosis(axis=1)
        )
        self.__application_train["NEW_LIVE_IND_SUM"] = (
            self.__application_train[
                ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]].sum(axis=1)
        )
        self.__application_train["NEW_LIVE_IND_KURT"] = (
            self.__application_train[
                ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]].kurtosis(axis=1)
        )
        self.__application_train["NEW_CONTACT_IND_SUM"] = (
            self.__application_train[
                ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
                 "FLAG_EMAIL"]].sum(axis=1)
        )
        self.__application_train["NEW_CONTACT_IND_KURT"] = (
            self.__application_train[
                ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",
                 "FLAG_EMAIL"]].kurtosis(axis=1)
        )

        # CNT_CHILDREN
        self.__application_train["NEW_INC_PER_CHLD"] = (
            # 分母 + 1 防止分母为 0
            self.__application_train["AMT_INCOME_TOTAL"] / self.__application_train["CNT_CHILDREN"].replace(0, np.nan)
        )

        # EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
        self.__application_train["NEW_SOURCES_PROD"] = (
            self.__application_train["EXT_SOURCE_1"] * self.__application_train["EXT_SOURCE_2"] * self.__application_train["EXT_SOURCE_3"]
        )
        self.__application_train["NEW_SOURCES_MEAN"] = (
            self.__application_train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].median(axis=1)
        )
        self.__application_train["NEW_SOURCES_STD"] = self.__application_train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
        self.__application_train["NEW_SOURCES_STD"] = self.__application_train["NEW_SOURCES_STD"].fillna(self.__application_train["NEW_SOURCES_STD"].median())
        self.__application_train["NEW_SOURCES_NA_SUM"] = (
            self.__application_train["EXT_SOURCE_1"].isna().astype(np.float64) +
            self.__application_train["EXT_SOURCE_2"].isna().astype(np.float64) +
            self.__application_train["EXT_SOURCE_3"].isna().astype(np.float64)
        )
        self.__application_train["NEW_SOURCES_MIN"] = self.__application_train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
        self.__application_train["NEW_SOURCES_MAX"] = self.__application_train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
        self.__application_train["NEW_SOURCES_SUM"] = self.__application_train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].sum(axis=1)

        # DAYS_EMPLOYED, DAYS_BIRTH, OWN_CAR_AGE DAYS_LAST_PHONE_CHANGE
        self.__application_train["NEW_EMPLOY_TO_BIRTH_RATIO"] = (
            # How many days before the application the person started current employment /
            # Client's age in days at the time of application
            # 工龄 / 年龄
            self.__application_train["DAYS_EMPLOYED"] / self.__application_train["DAYS_BIRTH"].replace(0, np.nan)
        )
        self.__application_train["NEW_CAR_TO_BIRTH_RATIO"] = (
            self.__application_train["OWN_CAR_AGE"] / self.__application_train["DAYS_BIRTH"].replace(0, np.nan)
        )
        self.__application_train["NEW_CAR_TO_EMPLOY_RATIO"] = (
            self.__application_train["OWN_CAR_AGE"] / self.__application_train["DAYS_EMPLOYED"].replace(0, np.nan)
        )
        self.__application_train["NEW_PHONE_TO_BIRTH_RATIO"] = (
            self.__application_train["DAYS_LAST_PHONE_CHANGE"] / self.__application_train["DAYS_BIRTH"].replace(0, np.nan)
        )
        self.__application_train["NEW_PHONE_TO_EMPLOY_RATIO"] = (
            self.__application_train["DAYS_LAST_PHONE_CHANGE"] / self.__application_train["DAYS_EMPLOYED"].replace(0, np.nan)
        )

        # REG_REGION_NOT_LIVE_REGION REG_REGION_NOT_WORK_REGION LIVE_REGION_NOT_WORK_REGION REG_CITY_NOT_LIVE_CITY
        # REG_CITY_NOT_WORK_CITY LIVE_CITY_NOT_WORK_CITY
        self.__application_train["NEW_REG_IND_SUM"] = (
            self.__application_train[
                ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
                 "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].sum(axis=1)
        )

        self.__application_train["NEW_REG_IND_KURT"] = (
            self.__application_train[
                ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
                 "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].kurtosis(axis=1)
        )

        # neptune.ml
        self.__application_train["NEW_CHILDREN_RATIO"] = (
            self.__application_train["CNT_CHILDREN"] / self.__application_train["CNT_FAM_MEMBERS"].replace(0, np.nan)
        )

        self.__application_train["NEW_INCOME_CREDIT_PERCENTAGE"] = self.__application_train["AMT_INCOME_TOTAL"] / self.__application_train["AMT_CREDIT"].replace(0, np.nan)

        self.__application_train["NEW_INCOME_PER_PERSON"] = self.__application_train["AMT_INCOME_TOTAL"] / self.__application_train["CNT_FAM_MEMBERS"].replace(0, np.nan)

        self.__application_train["NEW_PAYMENT_RATE"] = self.__application_train["AMT_ANNUITY"] / self.__application_train["AMT_CREDIT"].replace(0, np.nan)

        self.__application_train["NEW_CNT_NON_CHILD"] = self.__application_train["CNT_FAM_MEMBERS"] - self.__application_train["CNT_CHILDREN"]

        self.__application_train["NEW_CHILD_TO_NON_CHILD_RATIO"] = (
            self.__application_train["CNT_CHILDREN"] / self.__application_train["NEW_CNT_NON_CHILD"].replace(0, np.nan)
        )

        self.__application_train["NEW_INCOME_PER_NON_CHILD"] = self.__application_train["AMT_INCOME_TOTAL"] / self.__application_train["NEW_CNT_NON_CHILD"].replace(0, np.nan)

        self.__application_train["NEW_CREDIT_PER_PERSON"] = self.__application_train["AMT_CREDIT"] / self.__application_train["CNT_FAM_MEMBERS"].replace(0, np.nan)
        self.__application_train["NEW_CREDIT_PER_CHILD"] = self.__application_train["AMT_CREDIT"] / self.__application_train["CNT_CHILDREN"].replace(0, np.nan)
        self.__application_train["NEW_CREDIT_PER_NON_CHILD"] = self.__application_train["AMT_CREDIT"] / self.__application_train["NEW_CNT_NON_CHILD"].replace(0, np.nan)

        self.__application_train["NEW_RETIREMENT_AGE"] = (self.__application_train["DAYS_BIRTH"] < -14000).astype(int)
        self.__application_train["NEW_LONG_EMPLOYMENT"] = (self.__application_train["DAYS_EMPLOYED"] < -2000).astype(int)

        NEW_AGGREGATION_RECIPIES = [
            (["CODE_GENDER",
              "NAME_EDUCATION_TYPE"], [("AMT_ANNUITY", "max"),
                                       ("AMT_CREDIT", "max"),
                                       ("EXT_SOURCE_1", "median"),
                                       ("EXT_SOURCE_2", "median"),
                                       ("OWN_CAR_AGE", "max"),
                                       ("OWN_CAR_AGE", "sum"),
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
              "ORGANIZATION_TYPE"], [("AMT_ANNUITY", "median"),
                                     ("AMT_INCOME_TOTAL", "median"),
                                     ("DAYS_REGISTRATION", "median"),
                                     ("EXT_SOURCE_1", "median"),
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
              "REG_CITY_NOT_WORK_CITY"], [("AMT_ANNUITY", "median"),
                                          ("CNT_CHILDREN", "median"),
                                          ("DAYS_ID_PUBLISH", "median"),
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
              "REG_CITY_NOT_WORK_CITY"], [("EXT_SOURCE_1", "median"),
                                          ("EXT_SOURCE_2", "median"),
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
              "OCCUPATION_TYPE"], [("AMT_CREDIT", "median"),
                                   ("AMT_REQ_CREDIT_BUREAU_YEAR", "median"),
                                   ("APARTMENTS_AVG", "median"),
                                   ("BASEMENTAREA_AVG", "median"),
                                   ("EXT_SOURCE_1", "median"),
                                   ("EXT_SOURCE_2", "median"),
                                   ("EXT_SOURCE_3", "median"),
                                   ("NONLIVINGAREA_AVG", "median"),
                                   ("OWN_CAR_AGE", "median"),
                                   ("YEARS_BUILD_AVG", "median"),
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
              "REG_CITY_NOT_WORK_CITY"], [("ELEVATORS_AVG", "median"),
                                          ("EXT_SOURCE_1", "median"),
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

            (["OCCUPATION_TYPE"], [("AMT_ANNUITY", "median"),
                                   ("CNT_CHILDREN", "median"),
                                   ("CNT_FAM_MEMBERS", "median"),
                                   ("DAYS_BIRTH", "median"),
                                   ("DAYS_EMPLOYED", "median"),
                                   ("DAYS_ID_PUBLISH", "median"),
                                   ("DAYS_REGISTRATION", "median"),
                                   ("EXT_SOURCE_1", "median"),
                                   ("EXT_SOURCE_2", "median"),
                                   ("EXT_SOURCE_3", "median"),
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

        for groupby_cols, specs in NEW_AGGREGATION_RECIPIES:
            group_object = self.__application_train.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = "{}_{}_{}_{}".format("NEW", "_".join(groupby_cols), agg, select)
                self.__application_train = self.__application_train.merge(
                    group_object[select]
                        .agg(agg)
                        .reset_index()
                        .rename(index=str, columns={select: groupby_aggregate_name}),
                    left_on=groupby_cols,
                    right_on=groupby_cols,
                    how="left"
                )

    def data_return(self):
        # print(self.__application_train.iloc[0, :])

        return self.__application_train

if __name__ == "__main__":
    pat = PrepareApplicationTrain(
        input_path="C:\\Users\\puhui\\Desktop"
    )
    pat.data_prepare()
    pat.data_transform()
    pat.data_generate()
    pat.data_return()