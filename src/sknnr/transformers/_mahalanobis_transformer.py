import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from . import StandardScalerWithDOF


class MahalanobisTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.scaler_ = StandardScalerWithDOF(ddof=1).fit(X)
        covariance = np.cov(self.scaler_.transform(X), rowvar=False)
        self.transform_ = np.linalg.inv(np.linalg.cholesky(covariance).T)
        return self

    def transform(self, X, y=None):
        return self.scaler_.transform(X) @ self.transform_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
