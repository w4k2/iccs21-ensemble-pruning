"""
Stratified Bagging.
"""

from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import base
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from sklearn.utils.multiclass import unique_labels
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE

ba = balanced_accuracy_score


class StratifiedBagging(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator = None, ensemble_size=10, random_state=None, acc_prob=True, oversampled="None"):
        """Initialization."""
        # self._base_clf = base_estimator
        self.ensemble_size = ensemble_size
        self.estimators_ = []
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.acc_prob = acc_prob
        self.oversampled = oversampled

    # Fitting
    def fit(self, X, y):
        """Fitting."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        minority_X = self.X_[self.y_ == 1]
        minority_y = self.y_[self.y_ == 1]
        majority_X = self.X_[self.y_ == 0]
        majority_y = self.y_[self.y_ == 0]

        for i in range(self.ensemble_size):
            self.estimators_.append(base.clone(self.base_estimator))

        for n, estimator in enumerate(self.estimators_):
            np.random.seed(self.random_state + (n*2))
            bagXminority = minority_X[np.random.choice(round(minority_X.shape[0]/2), len(minority_y), replace=True), :]
            bagXmajority = majority_X[np.random.choice(round(majority_X.shape[0]/2), len(majority_y), replace=True), :]

            bagyminority = np.ones(len(minority_y)).astype('int')
            bagymajority = np.zeros(len(majority_y)).astype('int')

            train_X = np.concatenate((bagXmajority, bagXminority))
            train_y = np.concatenate((bagymajority, bagyminority))

            # unique, counts = np.unique(train_y, return_counts=True)

            if self.oversampled == "ROS":
                ovs = RandomOverSampler(random_state=self.random_state)
                train_X, train_y = ovs.fit_resample(train_X, train_y)
            elif self.oversampled == "SMOTE":
                ovs = SMOTE(random_state=self.random_state)
                train_X, train_y = ovs.fit_resample(train_X, train_y)
            elif self.oversampled == "SVMSMOTE":
                ovs = SVMSMOTE(random_state=self.random_state)
                train_X, train_y = ovs.fit_resample(train_X, train_y)
            elif self.oversampled == "B2SMOTE":
                ovs = BorderlineSMOTE(random_state=self.random_state, kind="borderline-2")
                train_X, train_y = ovs.fit_resample(train_X, train_y)

            estimator.fit(train_X, train_y)

        # Return the classifier
        return self

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.estimators_])

    def predict(self, X):
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")
        if self.acc_prob == True:
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return prediction
        else:
            pred_ = []
            # Modele w zespole dokonuja predykcji
            for member_clf in self.estimators_:
                pred_.append(member_clf.predict(X))
            # Zamiana na miacierz numpy (ndarray)
            pred_ = np.array(pred_)
            # Liczenie glosow
            prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
            return prediction

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)

        return average_support

    def score(self, X, y):
        return ba(y, self.predict(X))
