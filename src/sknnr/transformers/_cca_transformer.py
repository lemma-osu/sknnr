import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._cca import CCA


class CCATransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, spp=None):
        y = spp if spp is not None else y
        X, y = np.asarray(X), np.asarray(y)
        self.cca_ = CCA(X, y)
        return self

    def transform(self, X, y=None):
        X = X - self.cca_.env_center
        X = X @ self.cca_.coefficients
        return X @ self.cca_.axis_weights

    def fit_transform(self, X, y=None, spp=None):
        y = spp if spp is not None else y
        return self.fit(X, y).transform(X)
