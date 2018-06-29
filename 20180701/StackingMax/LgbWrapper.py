# coding:utf-8


class LgbWrapper(object):

    def __init__(self, *, clf, dataset_params, train_params, seed=7):
        self.__dataset_params = dataset_params
        self.__categorical_feature = self.__dataset_params.pop("categorical_feature")
        self.__train_params = train_params
        self.__train_params["feature_fraction_seed"] = seed  # feature_fraction like colsample_bytree
        self.__train_params["bagging_seed"] = seed  # 抽取部分样本, 注意不是重抽样, 也就是样本数量会变少
        self.__num_boost_round = self.__train_params.pop("num_boost_round")
        self.__lgb = clf
        self.__clf = None

    def train(self, train_feature, train_label):
        train = self.__lgb.Dataset(
            data=train_feature,
            label=train_label,
            feature_name=train_feature.columns.tolist(),
            categorical_feature=self.__categorical_feature.tolist()
        )
        self.__clf = self.__lgb.train(
            self.__train_params,
            train,
            num_boost_round=self.__num_boost_round,
            categorical_feature=self.__categorical_feature.tolist()
        )

    def predict(self, test_feature):
        return self.__clf.predict(test_feature.values)