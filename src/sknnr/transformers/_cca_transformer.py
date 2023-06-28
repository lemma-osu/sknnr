import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from . import ComponentReducerMixin
from ._cca import CCA


class CCATransformer(ComponentReducerMixin, TransformerMixin, BaseEstimator):
    @property
    def _n_features_out(self):
        return self.n_components_

    def get_feature_names_out(self) -> NDArray:
        return np.asarray(
            [f"cca{i}" for i in range(self._n_features_out)], dtype=object
        )

    def fit(self, X, y):
        self._validate_data(
            X, reset=True, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
        )

        X, y = np.asarray(X), np.asarray(y)
        self.cca_ = CCA(X, y)
        self.set_components(self.cca_)
        self.projector_ = self.cca_.projector(n_components=self.n_components_)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = np.asarray(X)
        return (X - self.cca_.env_center) @ self.projector_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
