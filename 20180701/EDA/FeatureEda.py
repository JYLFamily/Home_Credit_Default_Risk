# coding:utf-8

import gc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class FeatureEda(object):
    @staticmethod
    def target(*, df, target_name):
        df_mini = df[[target_name]].copy()
        temp_target = df_mini.groupby(target_name)[target_name].count()
        df_target = pd.DataFrame({"target": temp_target.index, "count": temp_target.values})

        # plot
        plt.subplots(figsize=(8, 6))
        plt.title("Distribution of %s" % target_name)
        sns.barplot(x="target", y="count", data=df_target)
        plt.tick_params(axis="both", which="major", labelsize=10)
        plt.show()
        plt.close()

        del df_mini, temp_target, df_target
        gc.collect()

    @staticmethod
    def missing(*, df, target_name):
        df_mini = df[[col for col in df.columns if col != target_name]].copy()
        temp_missing = (df_mini.isnull().sum()/df_mini.isnull().count() * 100).sort_values(ascending=False)
        df_percent = pd.DataFrame({"feature": temp_missing.index, "percent": temp_missing.values})

        # plot
        plt.subplots(figsize=(12, 6))
        plt.title("Missing data")
        s = sns.barplot(x="feature", y="percent", data=df_percent)
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
        plt.tick_params(axis="both", which="major", labelsize=8)
        plt.show()
        plt.close()

        del df_mini, temp_missing, df_percent
        gc.collect()

    @staticmethod
    def categorical_feature_eda(*, df, feature_name, target_name, label_rotation=False):
        df_mini = df[[feature_name, target_name]].copy()
        df_mini = df_mini.fillna("missing")

        # feature level
        temp_count = df_mini["CODE_GENDER"].value_counts()
        df_count = pd.DataFrame({"level": temp_count.index, "count": temp_count.values})

        # feature level - TARGET
        temp_mean = df_mini[["CODE_GENDER", "TARGET"]].groupby("CODE_GENDER")["TARGET"].mean()
        df_mean = pd.DataFrame({"level": temp_mean.index, "mean": temp_mean.values})

        # plot
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        s = sns.barplot(ax=ax1, x="level", y="count", data=df_count)
        s.set_xticklabels(s.get_xticklabels(), rotation=90) if label_rotation else s
        s = sns.barplot(ax=ax2, x="level", y="mean", data=df_mean)
        s.set_xticklabels(s.get_xticklabels(), rotation=90) if label_rotation else s
        plt.tick_params(axis="both", which="major", labelsize=10)
        plt.show()
        plt.close()

        del df_mini, temp_count, df_count, temp_mean, df_mean
        gc.collect()

    @staticmethod
    def numeric_feature_eda(*, df, feature_name, target_name):
        df_mini_target_pos = df.loc[(df[target_name] == 1), feature_name].copy()
        df_mini_target_neg = df.loc[(df[target_name] == 0), feature_name].copy()

        # plot
        plt.subplots(figsize=(12, 6))
        plt.title("Distribution of %s" % feature_name)
        sns.distplot(df_mini_target_pos, kde=False, label="TARGET = 1", norm_hist=True, bins=10)
        sns.distplot(df_mini_target_neg, kde=False, label="TARGET = 0", norm_hist=True, bins=10)
        plt.ylabel("frequency", fontsize=12)
        plt.xlabel(feature_name, fontsize=12)
        plt.tick_params(axis="both", which="major", labelsize=10)
        plt.legend()
        plt.show()
        plt.close()

        del df_mini_target_pos, df_mini_target_neg
        gc.collect()


if __name__ == "__main__":
    application_train = pd.read_csv("D:\\Kaggle\\Home_Credit_Default_Risk\\clean_data\\application_train.csv")
    FeatureEda.missing(df=application_train, target_name="TARGET")
    # FeatureEda.target(df=application_train, target_name="TARGET")
    # FeatureEda.categorical_feature_eda(df=application_train, feature_name="CODE_GENDER", target_name="TARGET")
    # FeatureEda.numeric_feature_eda(df=application_train, feature_name="DAYS_ID_PUBLISH", target_name="TARGET")
