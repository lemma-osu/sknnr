import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from ._cca import CCA


class CCATransformer(TransformerMixin, BaseEstimator):
    @property
    def _n_features_out(self):
        return self.cca_.eigenvalues.shape[0]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(
            [f"cca{i}" for i in range(self._n_features_out)], dtype=object
        )

    def fit(self, X, y):
        self._validate_data(
            X, reset=True, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
        )

        X, y = np.asarray(X), np.asarray(y)
        self.cca_ = CCA(X, y)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = np.asarray(X)

        X = X - self.cca_.env_center
        X = X @ self.cca_.coefficients
        return X @ self.cca_.axis_weights

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
