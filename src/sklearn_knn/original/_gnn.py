from sklearn.utils.validation import check_is_fitted

from .._base import IDNeighborsClassifier
from ._cca_transformer import CCATransformer


class GNN(IDNeighborsClassifier):
    def fit(self, X, y, spp=None):
        self.transform_ = CCATransformer().fit(X, y=y, spp=spp)
        X = self.transform_.transform(X)
        return super().fit(X, y)

    def predict(self, X):
        X = self.transform_.transform(X)
        return super().predict(X)

    def kneighbor_ids(self, X=None, n_neighbors=None):
        if X is not None:
            check_is_fitted(self, "transform_")
            X = self.transform_.transform(X)
        return super().kneighbor_ids(X=X, n_neighbors=n_neighbors)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is not None:
            check_is_fitted(self, "transform_")
            X = self.transform_.transform(X)
        return super().kneighbors(
            X=X, n_neighbors=n_neighbors, return_distance=return_distance
        )
