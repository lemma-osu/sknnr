from ._base import _TransformedKNeighborsRegressor
from .transformers import StandardScalerWithDOF


class EuclideanKNNRegressor(_TransformedKNeighborsRegressor):
    def fit(self, X, y):
        self.transform_ = StandardScalerWithDOF(ddof=1).fit(X)
        return super().fit(X, y)
