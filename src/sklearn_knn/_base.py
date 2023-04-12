from abc import abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y


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


class KnnPipeline(BaseEstimator):
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="distance",
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights

    @abstractmethod
    def _get_pipeline(self):
        raise NotImplementedError()

    def fit(self, X, y, **fit_params):
        if not hasattr(self, "_pipeline"):
            self._pipeline = self._get_pipeline()
        X, y = check_X_y(X, y)
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        self._pipeline.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self._pipeline.predict(X)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is None:
            return self._pipeline[-1].kneighbors(
                n_neighbors=n_neighbors, return_distance=return_distance
            )
        return self._pipeline[-1].kneighbors(
            X=self._pipeline[:-1].transform(X),
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )

    def kneighbor_ids(self, X=None, n_neighbors=None):
        if X is None:
            return self._pipeline[-1].kneighbor_ids(n_neighbors=n_neighbors)
        return self._pipeline[-1].kneighbor_ids(
            X=self._pipeline[:-1].transform(X),
            n_neighbors=n_neighbors,
        )
