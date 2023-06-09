from sklearn.neighbors import KNeighborsRegressor

from ._base import KNeighborsDFIndexCrosswalkMixin, TransformedKNeighborsMixin
from .transformers import MahalanobisTransformer


class MahalanobisKNNRegressor(
    KNeighborsDFIndexCrosswalkMixin, TransformedKNeighborsMixin, KNeighborsRegressor
):
    def fit(self, X, y):
        self.transform_ = MahalanobisTransformer().fit(X)
        return super().fit(X, y)
