from ._base import IDNeighborsRegressor, TransformedKNeighborsMixin
from .transformers import CCATransformer


class GNNRegressor(IDNeighborsRegressor, TransformedKNeighborsMixin):
    def fit(self, X, y, spp=None):
        self.transform_ = CCATransformer().fit(X, y=y, spp=spp)
        return super().fit(X, y)
