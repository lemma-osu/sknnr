from sklearn.neighbors import KNeighborsRegressor

from ._base import (
    IndependentPredictionMixin,
    KNeighborsDFIndexCrosswalkMixin,
    TransformedKNeighborsMixin,
)
from .transformers import StandardScalerWithDOF


class EuclideanKNNRegressor(
    KNeighborsDFIndexCrosswalkMixin,
    TransformedKNeighborsMixin,
    IndependentPredictionMixin,
    KNeighborsRegressor,
):
    def fit(self, X, y):
        self.transform_ = StandardScalerWithDOF(ddof=1).fit(X)
        return super().fit(X, y)
