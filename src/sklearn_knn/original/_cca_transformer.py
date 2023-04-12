import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._cca import CCA


class CCATransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, Y=None, spp=None):
        Y = spp if spp is not None else Y
        self.cca = CCA(X, Y)
        return self

    def transform(self, X, Y=None):
        X = X - self.cca.env_center
        X = np.dot(X, self.cca.coefficients)
        return np.dot(X, self.cca.axis_weights)

    def fit_transform(self, X, Y=None, spp=None):
        Y = spp if spp is not None else Y
        return self.fit(X, Y).transform(X)
