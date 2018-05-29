# coding:utf-8


class CatWrapper(object):

    def __init__(self, *, clf, init_params=None, train_params=None, seed=7):
        self.__init_params = init_params
        self.__init_params["random_seed"] = seed
        self.__train_params = train_params
        self.__clf = clf(** self.__init_params)

    def train(self, train_feature, train_label):
        self.__clf.fit(train_feature, train_label, ** self.__train_params)

    def predict(self, test_feature):
        return self.__clf.predict_proba(test_feature)[:, 1]