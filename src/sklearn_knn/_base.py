import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class IDNeighborsClassifier(KNeighborsClassifier):
    """
    Specialized KNeighborsClassifier where labels
    are IDs for samples and not classes
    """
    def kneighbor_ids(self, X=None, n_neighbors=None):
        neigh_ind = super().kneighbors(X, n_neighbors, False)
        return self.classes_[self._y[neigh_ind]]


class MyStandardScaler(StandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        sc = super().fit(X, y, sample_weight)
        sc.scale_ = np.std(X, axis=0, ddof=1)
        return sc
