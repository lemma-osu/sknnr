import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from . import ComponentReducerMixin
from ._cca import CCA


class CCATransformer(ComponentReducerMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        self._validate_data(
            X, reset=True, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
        )

        X, y = np.asarray(X), np.asarray(y)
        self.ordination_ = CCA(X, y)
        self.set_n_components()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = np.asarray(X)
        return (X - self.ordination_.env_center) @ self.ordination_.projector(
            n_components=self.n_components_
        )

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
