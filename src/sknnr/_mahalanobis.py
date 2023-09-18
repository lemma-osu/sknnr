from ._base import _TransformedKNeighborsRegressor
from .transformers import MahalanobisTransformer


class MahalanobisKNNRegressor(_TransformedKNeighborsRegressor):
    def fit(self, X, y):
        self.transform_ = MahalanobisTransformer().fit(X)
        return super().fit(X, y)
