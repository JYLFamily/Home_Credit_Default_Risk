# coding:utf-8

import os
import sys
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import Imputer
from gplearn.genetic import SymbolicTransformer
np.random.seed(7)


class GplearnGenerateFeature(object):
    def __init__(self, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # data prepare
        self.__feature_importance = None
        self.__feature_top_column = None
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_label = None
        self.__train_feature, self.__test_feature = [None for _ in range(2)]

        self.__categorical_columns = None
        self.__encoder = None
        self.__numeric_columns = None
        self.__filler = None

        # feature generate
        self.__genetic_transformer = None
        self.__genetic_train_feature = None
        self.__genetic_test_feature = None

    def data_prepare(self):
        self.__feature_importance = pd.read_csv(os.path.join(self.__input_path, "feature_importance_feature_data_V5.csv"))
        self.__feature_importance = (
            self.__feature_importance.groupby(["feature"])["importance"].mean().to_frame("importance").reset_index(drop=False)
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        self.__feature_top_column = list(self.__feature_importance.iloc[0:200, 0])

        self.__train = pd.read_csv(
            os.path.join(self.__input_path, "train_select_feature_df.csv"),
            usecols=self.__feature_top_column+["TARGET"]
        )
        self.__test = pd.read_csv(
            os.path.join(self.__input_path, "test_select_feature_df.csv"),
            usecols=self.__feature_top_column
        )

        self.__train_label = self.__train["TARGET"]
        self.__train_feature = self.__train.drop("TARGET", axis=1)
        self.__test_feature = self.__test[self.__train_feature.columns.tolist()]

        # encoder
        self.__categorical_columns = self.__train_feature.select_dtypes(include="object").columns.tolist()
        self.__encoder = TargetEncoder()
        self.__encoder.fit(self.__train_feature[self.__categorical_columns], self.__train_label)
        self.__train_feature[self.__categorical_columns] = self.__encoder.transform(
            self.__train_feature[self.__categorical_columns]
        )
        self.__test_feature[self.__categorical_columns] = self.__encoder.transform(
            self.__test_feature[self.__categorical_columns]
        )

        # filler
        self.__numeric_columns = self.__train_feature.select_dtypes(exclude="object").columns.tolist()
        self.__filler = Imputer(strategy="median")
        self.__filler.fit(
            self.__train_feature[self.__numeric_columns]
        )
        self.__train_feature[self.__numeric_columns] = self.__filler.transform(
            self.__train_feature[self.__numeric_columns]
        )
        self.__test_feature[self.__numeric_columns] = self.__filler.transform(
            self.__test_feature[self.__numeric_columns]
        )

    def feature_generate(self):
        self.__genetic_transformer = SymbolicTransformer(
            population_size=10000,
            generations=200,
            tournament_size=200,
            metric="spearman",
            n_jobs=-1,
            verbose=1
        )
        self.__genetic_transformer.fit(
            self.__train_feature, self.__train_label
        )
        self.__genetic_train_feature = self.__genetic_transformer.transform(self.__train_feature)
        self.__genetic_test_feature = self.__genetic_transformer.transform(self.__test_feature)

    def data_output(self):
        self.__genetic_train_feature = pd.DataFrame(
            self.__genetic_train_feature,
            columns=["Genetic_" + str(i) for i in range(self.__genetic_train_feature.shape[1])]
        )
        self.__genetic_test_feature = pd.DataFrame(
            self.__genetic_test_feature,
            columns=["Genetic_" + str(i) for i in range(self.__genetic_test_feature.shape[1])]
        )
        self.__genetic_train_feature.to_csv(os.path.join(self.__output_path, "genetic_train_feature.csv"), index=False)
        self.__genetic_test_feature.to_csv(os.path.join(self.__output_path, "genetic_test_feature.csv"), index=False)


if __name__ == "__main__":
    ggf = GplearnGenerateFeature(
        input_path=sys.argv[1],
        output_path=sys.argv[2]
    )
    ggf.data_prepare()
    ggf.feature_generate()
    ggf.data_output()