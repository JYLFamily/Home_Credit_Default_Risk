# coding:utf-8


class LgBoostWrapper(object):

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
        train = self.__lgb.DataSet(
            data=train_feature,
            label=train_label,
            feature_name=train_feature.columns,
            categorical_feature=self.__categorical_feature
        )
        self.__clf.fit(self.__train_params, train, num_boost_round=self.__num_boost_round)

    def predict(self, test_feature):
        test = self.__lgb.DataSet(
            data=test_feature,
            feature_names=test_feature.columns,
            categorical_feature=self.__categorical_feature
        )
        return self.__clf.predict_proba(test)[:, 1]