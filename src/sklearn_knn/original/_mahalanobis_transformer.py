import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .._base import MyStandardScaler


class MahalanobisTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.scalar_ = MyStandardScaler().fit(X)
        covariance = np.cov(self.scalar_.transform(X), rowvar=False)
        self.transform_ = np.linalg.inv(np.linalg.cholesky(covariance).T)
        return self

    def transform(self, X, y=None):
        return self.scalar_.transform(X) @ self.transform_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
