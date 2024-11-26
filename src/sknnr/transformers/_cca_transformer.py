import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from . import ComponentReducerMixin
from ._cca import CCA


class CCATransformer(ComponentReducerMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        X = self._validate_data(
            X,
            reset=True,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
            ensure_min_features=2,
            ensure_min_samples=1,
        )
        y = np.asarray(y)
        if len(y.shape) < 2:
            raise ValueError("`y` must be a 2D array.")

        self.ordination_ = CCA(X, y)
        self.set_n_components()
        self.env_center_ = self.ordination_.env_center
        self.projector_ = self.ordination_.projector(n_components=self.n_components_)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = self._validate_data(
            X,
            reset=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
            ensure_min_features=2,
            ensure_min_samples=1,
        )
        return (X - self.env_center_) @ self.projector_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {
            "allow_nan": False,
            "requires_fit": True,
            "requires_y": True,
        }
