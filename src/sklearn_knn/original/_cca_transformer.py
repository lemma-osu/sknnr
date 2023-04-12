import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._cca import CCA


class CCATransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, spp=None):
        y = spp if spp is not None else y
        self.cca = CCA(X, y)
        return self

    def transform(self, X, y=None):
        X = X - self.cca.env_center
        X = np.dot(X, self.cca.coefficients)
        return np.dot(X, self.cca.axis_weights)

    def fit_transform(self, X, y=None, spp=None):
        y = spp if spp is not None else y
        return self.fit(X, y).transform(X)
