import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._cca import CCA


class CCATransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, Y=None, **fit_params):
        Y = fit_params.get("spp", Y)
        self.cca = CCA(X, Y)
        return self

    def transform(self, X, Y=None):
        X = X - self.cca.env_center
        X = np.dot(X, self.cca.coefficients)
        return np.dot(X, self.cca.axis_weights)

    def fit_transform(self, X, Y=None, **fit_params):
        Y = fit_params.get("spp", Y)
        return self.fit(X, Y).transform(X)
