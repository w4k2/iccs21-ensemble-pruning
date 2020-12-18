import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics import accuracy_score
from utils import calc_diversity_measures, calc_diversity_measures2
from sklearn.cluster import KMeans
from strlearn.metrics import balanced_accuracy_score, recall, precision
import seaborn as sns
from scipy import stats


class CMOE(BaseEnsemble, ClassifierMixin):
    """
    Clustering-based ensemble pruning for ransom subspace
    """

    def __init__(self, base_estimator=None, diversity=None, max_clusters=7, hard_voting=False, random_state=None):
        self.base_estimator = base_estimator
        self.diversity = diversity
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.hard_voting = hard_voting
        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        self.X_, self.y_ = X, y

        # Base clf, ensemble and subspaces
        self.clf_ = clone(self.base_estimator).fit(self.X_, self.y_)
        self.ensemble_ = self.clf_.estimators_
        # self.subspaces_ = self.clf_.subspaces

        # Calculate mean accuracy on training set
        p = np.mean(np.array([ accuracy_score(self.y_,member_clf.predict(self.X_)) for clf_ind, member_clf in enumerate(self.ensemble_)]))

        # All measures for whole ensemble
        self.e, self.k, self.kw, self.dis, self.q = calc_diversity_measures(self.X_, self.y_, self.ensemble_, p)

        # Calculate diversity space for all measures
        self.diversity_space = np.zeros((5, len(self.ensemble_)))
        for i in range(len(self.ensemble_)):
            temp_ensemble = self.ensemble_.copy()
            temp_ensemble.pop(i)
            # temp_subspaces = self.subspaces_[np.arange(len(self.subspaces_))!=1]

            p = np.mean(np.array([ accuracy_score(self.y_,member_clf.predict(self.X_)) for clf_ind, member_clf in enumerate(self.ensemble_)]))


            temp_e, temp_k, temp_kw, temp_dis, temp_q = calc_diversity_measures(self.X_, self.y_, temp_ensemble, p)
            self.diversity_space[0,i] = self.e - temp_e
            self.diversity_space[1,i] = self.k - temp_k
            self.diversity_space[2,i] = self.kw - temp_kw
            self.diversity_space[3,i] = self.dis - temp_dis
            self.diversity_space[4,i] = self.q - temp_q

        # Clustering
        # DIV x CLUSTERS x CLFS
        self.indexes = np.zeros((5, self.max_clusters-1, len(self.ensemble_)))

        for div_inxd, div in enumerate(self.diversity_space):
            for clu_indx, n_clusters in enumerate(range(2, self.max_clusters+1)):
                self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                self.indexes[div_inxd, clu_indx] = self.kmeans.fit_predict(div.reshape(-1,1))

        ensemble_ = np.array(self.ensemble_)
        self.all_clusters_ensembles = []
        for div_inxd in range(5):
            for cluster_indx, n_clusters in enumerate(range(2, self.max_clusters+1)):
                self.cluster_ensembles = []
                self.mor_ensemble = []
                for j in range(n_clusters):
                    cluster_ensemble = ensemble_[self.indexes[div_inxd,cluster_indx]==j]
                    self.cluster_ensembles.append(cluster_ensemble)

                # print(len(self.cluster_ensembles))

                for repeat in range(len(self.cluster_ensembles)):
                    mor_ensemble = []
                    for ens in range(len(self.cluster_ensembles)):
                        # print(len(self.cluster_ensembles[ens]))
                        selected_model = np.random.randint(0, len(self.cluster_ensembles[ens]))
                        # print("Wybralem z %i base %i" %(ens, selected_model))
                        mor_ensemble.append(self.cluster_ensembles[ens][selected_model])
                    self.mor_ensemble.append(mor_ensemble)
                self.all_clusters_ensembles.append(np.array(self.mor_ensemble))
                # self.pruned_ensembles.append(self.pruned_ensemble_)
                # print(len(self.pruned_ensemble_))
                    # self.pruned_subspaces_.append(cluster_subspaces[best])
                # print(len(self.pruned_ensemble_))
            # exit()
            # exit()
        self.all_clusters_ensembles = np.array(self.all_clusters_ensembles)
        # print(self.all_clusters_ensembles.shape)

        # exit()
        # print(np.array(self.pruned_ensembles).shape)
        # exit()

        return self


    # def ensemble_support_matrix(self, X):
    #     return np.array(
    #     [member_clf.predict_proba(X) for member_clf in self.pruned_ensemble_]
    #     )
    #
    # def ensemble_support_matrix2(self, X):
    #     return np.array(
    #     [[member_clf.predict_proba(X) for member_clf in pruned_ensemble] for pruned_ensemble in self.pruned_ensembles]
    #     )
    #
    # def ensemble_support_matrix3(self, X):
    #     for m, method in enumerate(self.all_clusters_ensembles):
    #         print(method)
    #         exit()
    #     exit()
    #     return False
    #
    # def predict_proba(self, X):
    #     esm = self.ensemble_support_matrix(X)
    #     average_support = np.mean(esm, axis=0)
    #     return average_support

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")
        all_preds = np.zeros((30, X.shape[0]))
        # All 30 methods
        for m, method in enumerate(self.all_clusters_ensembles):
            # Preds for each cluster in method
            ensemble_votes = []
            for e, ensemble in enumerate(method):
                # print(ensemble.shape)
            # exit()




# """
                cluster_preds = np.array([clf.predict(X) for clf in ensemble])
                # print(cluster_preds, cluster_preds.shape)
                mv1, _ = stats.mode(cluster_preds, axis=0)
                # print(mv1, mv1.shape)
                ensemble_votes.append(mv1)
            ensemble_votes = np.array(ensemble_votes)
            mv2, _ = stats.mode(ensemble_votes, axis=0)
            mv2 = np.squeeze(mv2)
            all_preds[m] = mv2.astype(int)
        # print(all_preds.astype(int), all_preds.shape)
        # exit()
        return all_preds.astype(int)
# """
