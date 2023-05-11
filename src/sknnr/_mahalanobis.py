from sklearn.utils.validation import check_is_fitted

from ._base import IDNeighborsRegressor, TransformedKNeighborsMixin
from .transformers import MahalanobisTransformer


class MahalanobisKNNRegressor(IDNeighborsRegressor, TransformedKNeighborsMixin):
    def fit(self, X, y):
        self.transform_ = MahalanobisTransformer().fit(X)
        X = self.transform_.transform(X)
        return super().fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = self.transform_.transform(X)
        return super().predict(X)
