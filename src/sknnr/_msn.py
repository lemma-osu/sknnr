from sklearn.neighbors import KNeighborsRegressor

from ._base import KNeighborsDFIndexCrosswalkMixin, TransformedKNeighborsMixin
from .transformers import CCorATransformer


class MSNRegressor(
    KNeighborsDFIndexCrosswalkMixin, TransformedKNeighborsMixin, KNeighborsRegressor
):
    def __init__(self, n_components=None, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

    def fit(self, X, y, y_fit=None):
        y_fit = y_fit if y_fit is not None else y
        self.transform_ = CCorATransformer(self.n_components).fit(X, y=y_fit)
        return super().fit(X, y)
