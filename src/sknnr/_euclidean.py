from sklearn.neighbors import KNeighborsRegressor

from ._base import KNeighborsDFIndexCrosswalkMixin, TransformedKNeighborsMixin
from .transformers import StandardScalerWithDOF


class EuclideanKNNRegressor(
    KNeighborsDFIndexCrosswalkMixin, TransformedKNeighborsMixin, KNeighborsRegressor
):
    def fit(self, X, y):
        self.transform_ = StandardScalerWithDOF(ddof=1).fit(X)
        return super().fit(X, y)
