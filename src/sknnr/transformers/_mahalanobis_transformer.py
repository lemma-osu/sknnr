import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import StandardScalerWithDOF


class MahalanobisTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self._validate_data(X, force_all_finite="allow-nan", reset=True)

        self.scaler_ = StandardScalerWithDOF(ddof=1).fit(X)
        covariance = np.cov(self.scaler_.transform(X), rowvar=False)
        self.transform_ = np.linalg.inv(np.linalg.cholesky(covariance).T)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        self._validate_data(X, force_all_finite="allow-nan", reset=False)

        return self.scaler_.transform(X) @ self.transform_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"allow_nan": True}
