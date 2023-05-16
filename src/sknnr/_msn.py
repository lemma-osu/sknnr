from ._base import IDNeighborsRegressor, TransformedKNeighborsMixin
from .transformers import CCorATransformer


class MSNRegressor(IDNeighborsRegressor, TransformedKNeighborsMixin):
    def fit(self, X, y, spp=None):
        self.transform_ = CCorATransformer().fit(X, y=y, spp=spp)
        return super().fit(X, y)
