# coding:utf-8


class XgbWrapper(object):

    def __init__(self, *, clf, train_params=None, seed=7):
        self.__train_params = train_params
        self.__train_params["seed"] = seed
        self.__num_boost_round = self.__train_params.pop("num_boost_round")
        self.__xgb = clf
        self.__clf = None

    def train(self, train_feature, train_label):
        train = self.__xgb.DMatrix(
            data=train_feature,
            label=train_label,
            feature_names=train_feature.columns
        )
        self.__clf = self.__xgb.train(self.__train_params, train, self.__num_boost_round)

    def predict(self, test_feature):
        test = self.__xgb.DMatrix(
            data=test_feature,
            feature_names=test_feature.columns
        )
        return self.__clf.predict(test)