from ._base import IDNeighborsRegressor, TransformedKNeighborsMixin
from .transformers import MahalanobisTransformer


class MahalanobisKNNRegressor(IDNeighborsRegressor, TransformedKNeighborsMixin):
    def fit(self, X, y):
        self.transform_ = MahalanobisTransformer().fit(X)
        return super().fit(X, y)
