from sklearn.base import BaseEstimator, TransformerMixin

from . import StandardScalerWithDOF
from ._ccora import CCorA


class CCorATransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, spp=None):
        y = spp if spp is not None else y
        self.scaler_ = StandardScalerWithDOF(ddof=1).fit(X)
        y = StandardScalerWithDOF(ddof=1).fit_transform(y)
        self.ccora_ = CCorA(self.scaler_.transform(X), y)
        return self

    def transform(self, X, y=None):
        return self.scaler_.transform(X) @ self.ccora_.projector

    def fit_transform(self, X, y=None, spp=None):
        y = spp if spp is not None else y
        return self.fit(X, y).transform(X)
