import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class IDNeighborsRegressor(KNeighborsRegressor):
    """
    Placeholder class for implementing plot ID access.
    """


class TransformedKNeighborsMixin(KNeighborsRegressor):
    """
    Mixin for KNeighbors regressors that store a `transform_` during fitting (e.g.
    GNN).
    """

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is not None:
            check_is_fitted(self, "transform_")
            X = self.transform_.transform(X)
        return super().kneighbors(
            X=X, n_neighbors=n_neighbors, return_distance=return_distance
        )


class MyStandardScaler(StandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        sc = super().fit(X, y, sample_weight)
        sc.scale_ = np.std(X, axis=0, ddof=1)
        return sc
