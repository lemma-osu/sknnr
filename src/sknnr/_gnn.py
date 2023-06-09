from sklearn.neighbors import KNeighborsRegressor

from ._base import KNeighborsDFIndexCrosswalkMixin, TransformedKNeighborsMixin
from .transformers import CCATransformer


class GNNRegressor(
    KNeighborsDFIndexCrosswalkMixin, TransformedKNeighborsMixin, KNeighborsRegressor
):
    def fit(self, X, y, y_fit=None):
        y_fit = y_fit if y_fit is not None else y
        self.transform_ = CCATransformer().fit(X, y=y_fit)
        return super().fit(X, y)
