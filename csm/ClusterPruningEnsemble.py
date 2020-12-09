import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics import accuracy_score
from utils import calc_diversity_measures, calc_diversity_measures2
from sklearn.cluster import KMeans
from strlearn.metrics import balanced_accuracy_score, recall, precision


class ClusterPruningEnsemble(BaseEnsemble, ClassifierMixin):
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

        # import matplotlib.pyplot as plt
        # import matplotlib as mplt
        # mplt.rcParams['axes.spines.right'] = False
        # mplt.rcParams['axes.spines.top'] = False
        # mplt.rcParams['axes.spines.left'] = False
        # plt.figure(figsize=(8,1))
        # plt.ylim(0, 0.2)
        # plt.yticks([])
        # # plt.xlim(-0.125, 0.075)
        # plt.tight_layout()
        # plt.vlines(0.05*self.diversity_space[1], 0, .2, color=(0.6015625,0.203125,0.17578125))
        # plt.savefig("foo.png")
        # # plt.show()
        # exit()

        # Clustering
        # DIV x CLUSTERS x CLFS
        self.indexes = np.zeros((5, self.max_clusters-1, len(self.ensemble_)))

        for div_inxd, div in enumerate(self.diversity_space):
            for clu_indx, n_clusters in enumerate(range(2, self.max_clusters+1)):
                self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                self.indexes[div_inxd, clu_indx] = self.kmeans.fit_predict(div.reshape(-1,1))
                # print(div_inxd, clu_indx)
                # print(self.indexes[div_inxd, clu_indx])

            # Plots
            # import matplotlib.pyplot as plt
            # import matplotlib as mplt
            # mplt.rcParams['axes.spines.right'] = False
            # mplt.rcParams['axes.spines.top'] = False
            # mplt.rcParams['axes.spines.left'] = False
            # plt.figure(figsize=(8,1))
            # plt.ylim(0, 0.2)
            # plt.yticks([])
            # # plt.xlim(-0.125, 0.075)
            # plt.tight_layout()
            # colors = ["red", "blue", "green", "orange", "cyan", "pink", "black", "yellow"]
            # for j in range(self.max_clusters):
            #     plt.vlines(0.05*self.diversity_space[0][self.kmeans.labels_ == j], 0, .2, color=colors[j])
            # plt.savefig("foo.png")

        # Calculate base models bac
        base_scores = np.array([ balanced_accuracy_score(self.y_,member_clf.predict(self.X_)) for clf_ind, member_clf in enumerate(self.ensemble_)])
        # print(base_scores)
        # exit()
        # DIV x CLU x CLU
        # self.pruned_ensembles = np.zeros((5, self.max_clusters-1, self.max_clusters-1))
        self.pruned_ensembles = []
        # self.pruned_subspaces_ = []
        ensemble_ = np.array(self.ensemble_)

        for div_inxd in range(5):
            for cluster_indx, n_clusters in enumerate(range(2, self.max_clusters+1)):
                self.pruned_ensemble_ = []
                for j in range(n_clusters):
                    cluster_ensemble = ensemble_[self.indexes[div_inxd,cluster_indx]==j]
                    # cluster_subspaces = self.subspaces_[indexes==j]
                    # print(cluster_ensemble.shape)
                    cluster_scores = base_scores[self.indexes[div_inxd,cluster_indx]==j]
                    # print(cluster_scores)
                    best = np.argmax(cluster_scores)
                    # print(best)
                    # exit()
                    self.pruned_ensemble_.append(cluster_ensemble[best])
                self.pruned_ensembles.append(self.pruned_ensemble_)
                # print(len(self.pruned_ensemble_))
                    # self.pruned_subspaces_.append(cluster_subspaces[best])
                # print(len(self.pruned_ensemble_))
        self.pruned_ensembles = np.array(self.pruned_ensembles)
        # print(np.array(self.pruned_ensembles).shape)
        # exit()

        # Single measures
        """
        # Calculate chosen diversity measure for whole ensemble
        self.whole_diversity = calc_diversity_measures2(self.X_, self.y_, self.ensemble_, self.subspaces_, p, self.diversity)

        # Calculate diversity space
        self.diversity_space = np.zeros((len(self.ensemble_)))
        for i in range(len(self.ensemble_)):
            temp_ensemble = self.ensemble_.copy()
            temp_ensemble.pop(i)
            temp_subspaces = self.subspaces_[np.arange(len(self.subspaces_))!=1]

            if self.diversity == "k":
                p = np.mean(np.array([ accuracy_score(self.y_,member_clf.predict(X[:, self.subspaces_[clf_ind]])) for clf_ind, member_clf in enumerate(self.ensemble_)]))

            temp_diversity_space = self.whole_diversity - calc_diversity_measures2(self.X_, self.y_, temp_ensemble, temp_subspaces, p, self.diversity)
            self.diversity_space[i] = temp_diversity_space

        import matplotlib.pyplot as plt
        import matplotlib as mplt
        mplt.rcParams['axes.spines.right'] = False
        mplt.rcParams['axes.spines.top'] = False
        mplt.rcParams['axes.spines.left'] = False
        plt.figure(figsize=(8,1))
        plt.ylim(0, 0.2)
        plt.yticks([])
        # plt.xlim(-0.125, 0.075)
        plt.tight_layout()
        plt.vlines(0.05*self.diversity_space, 0, .2, color=(0.6015625,0.203125,0.17578125))
        plt.savefig("foo.png")
        # plt.show()
        exit()
        """

        return self


    def ensemble_support_matrix(self, X):
        return np.array(
        [member_clf.predict_proba(X) for member_clf in self.pruned_ensemble_]
        )

    def ensemble_support_matrix2(self, X):
        return np.array(
        [[member_clf.predict_proba(X) for member_clf in pruned_ensemble] for pruned_ensemble in self.pruned_ensembles]
        )

    def predict_proba(self, X):
        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")

        if self.hard_voting == True:
            # Podejmowanie decyzji na podstawie twardego glosowania
            pred_ = []
            # Modele w zespole dokonuja predykcji
            for i, member_clf in enumerate(self.pruned_ensemble_):
                pred_.append(member_clf.predict(X))
            # Zamiana na miacierz numpy (ndarray)
            pred_ = np.array(pred_)
            # Liczenie glosow
            prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
            # Zwrocenie predykcji calego zespolu
            return self.classes_[prediction]
        else:
            preds = np.zeros((30, X.shape[0]))
            esm = self.ensemble_support_matrix2(X)
            for e_n, ensemble_esm in enumerate(esm):
                ensemble_esm = np.array(ensemble_esm)
                average_support = np.mean(ensemble_esm, axis=0)
                preds[e_n] = np.argmax(average_support, axis=1)
            return preds.astype(int)
