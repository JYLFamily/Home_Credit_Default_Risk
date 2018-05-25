# coding:utf-8


class SklWrapper(object):

    def __init__(self, *, clf, init_params=None, seed=7):
        self.__init_params = init_params
        self.__init_params["random_state"] = seed
        self.__clf = clf(** init_params)

    def train(self, train_feature, train_label):
        self.__clf.fit(train_feature, train_label)

    def predict(self, test_feature):
        return self.__clf.predict_proba(test_feature)[:, 1]

