import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from RF.feature_extraction import extract_all_features
from abc import ABC, abstractmethod

sys.path.append('../')


class Clf(ABC):

    def __init__(self, dga_type, dga):
        self.clf = None
        self.clf_type = ''
        self.param_map = {}
        self.dga_type = dga_type
        self.dga = dga

    @abstractmethod
    def predict(self, test, labels):
        pass


class RFClassifier(Clf):

    def __init__(self, dga='mix', criterion='gini', max_features='auto', n_estimators=20, min_samples_split=2,
                 max_depth=20, params=None):
        # super(RFClassifier, self).__init__(DGA_TYPE_MAP[dga], dga)
        super(RFClassifier, self).__init__('mix', dga)
        self.param_map = {'mix': {'criterion': criterion, 'max_features': max_features, 'n_estimators': n_estimators,
                                  'min_samples_split': min_samples_split, 'max_depth': max_depth}}
        if params:
            self.clf = RandomForestClassifier(**params)
        else:
            self.clf = RandomForestClassifier(**self.param_map['mix'])
        self.clf_type = 'rf'

    def training(self, train, labels):
        """
        Training on given data.
        :param train: array-like containing domain name strings
        :param labels: array-like containing labels
        :return: void
        """

        feature_matrix = extract_all_features(train)
        print('training feature matrix', len(feature_matrix[0]))
        self.clf.fit(feature_matrix, labels)

    def predict(self, test, labels=None):
        """
        Predict test data
        :param test: array of samples to predict
        :return: array of true labels, array of predicted labels
        """

        feature_matrix = extract_all_features(test)
        prediction = self.clf.predict(feature_matrix)

        if labels is not None:
            return labels, prediction
        else:
            return prediction

    def predict_proba(self, test, labels=None):
        """
        Predict test data
        :param test: array of samples to predict
        :return: array of true labels, array of predicted labels
        """
        feature_matrix = extract_all_features(test)
        prediction = self.clf.predict_proba(feature_matrix)

        if labels is not None:
            return labels, prediction
        else:
            return prediction

    def predict_bulk_preprocessed(self, feature_matrix):
        return self.clf.predict(feature_matrix)

    def cv(self, train, label):
        feature_matrix = extract_all_features(train)
        clf_s = cross_val_score(self.clf, feature_matrix, label, cv=10)
        return clf_s
