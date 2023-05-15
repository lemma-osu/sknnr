from ._base import IDNeighborsRegressor, TransformedKNeighborsMixin
from .transformers import StandardScalerWithDOF


class EuclideanKNNRegressor(IDNeighborsRegressor, TransformedKNeighborsMixin):
    def fit(self, X, y):
        self.transform_ = StandardScalerWithDOF(ddof=1).fit(X)
        return super().fit(X, y)
