import numpy as np
import weles as ws
from math import sqrt
from sklearn.base import clone
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
    precision,
    recall,
    specificity,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from csm import RandomSubspaceEnsemble, CMOE, StratifiedBagging
import multiprocessing


metrics = {
    "BAC": balanced_accuracy_score,
    "G-mean": geometric_mean_score_1,
    "F1": f1_score,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity
}

dataset_names = np.load("dataset_names_knn.npy").tolist()
# print(dataset_names)
# exit()

# data = ws.utils.Data(selection=("all", ["imbalanced"]), path="../datasets/")
data = ws.utils.Data(selection=dataset_names, path="../datasets/")
datasets = data.load()
# exit()
# for d_indx, (key, data) in enumerate(datasets.items()):
def worker(d_indx, key, data):
    print("Dataset: %s start" % key)
    X, y = data


    # b = GaussianNB()
    # b = DecisionTreeClassifier(random_state=1410)
    b = KNeighborsClassifier(weights='distance')
    # b = MLPClassifier(random_state=1410)
    # b = SVC(kernel='rbf', probability=False, random_state=1410)
    n_estimators = 50
    acc_prob = False

    base_clf = StratifiedBagging(base_estimator = b, ensemble_size=n_estimators, acc_prob=True, random_state=1410)

    clfs = {
        "MV-kNN-ROS": StratifiedBagging(base_estimator = b, ensemble_size=n_estimators, acc_prob=True, random_state=1410, oversampled="ROS"),
        "MV-kNN-SMOTE": StratifiedBagging(base_estimator = b, ensemble_size=n_estimators, acc_prob=True, random_state=1410, oversampled="SMOTE"),
        "MV-kNN-SVMSMOTE": StratifiedBagging(base_estimator = b, ensemble_size=n_estimators, acc_prob=True, random_state=1410, oversampled="SVMSMOTE"),
        "MV-kNN-B2SMOTE": StratifiedBagging(base_estimator = b, ensemble_size=n_estimators, acc_prob=True, random_state=1410, oversampled="B2SMOTE"),
    }

    n_splits = 5
    n_repeats = 1
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1410)
    scores = np.zeros((len(clfs), n_splits * n_repeats, len(metrics)))

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        print("FOLD %i: " % fold_id)
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf = clfs[clf_name]
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            for m_indx, (name, metric) in enumerate(metrics.items()):
                scores[clf_id, fold_id, m_indx] = metric(y[test], y_pred)
    np.save("results/knn/%s_knn_preproc" % key, scores)
    print("Dataset: %s end" % key)
# print(scores)
# print(np.mean(scores, axis=1))


jobs = []
for d_indx, (key, data) in enumerate(datasets.items()):
    p = multiprocessing.Process(target=worker, args=(d_indx, key, data))
    jobs.append(p)
    p.start()
