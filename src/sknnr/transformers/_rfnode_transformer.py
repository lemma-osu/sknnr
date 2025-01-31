from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .._base import _validate_data


class RFNodeTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators_per_forest: int = 50,
        max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
    ):
        self.n_estimators_per_forest = n_estimators_per_forest
        self.max_features = max_features

    def fit(self, X, y=None):
        _, y = _validate_data(self, X=X, y=y, reset=True, multi_output=True)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.rfs_ = [
            RandomForestRegressor(
                n_estimators=self.n_estimators_per_forest,
                max_features=self.max_features,
                random_state=42,
                min_samples_leaf=5,
            ).fit(X, y[:, i])
            for i in range(y.shape[1])
        ]
        return self

    def get_feature_names_out(self) -> NDArray:
        check_is_fitted(self, "rfs_")
        return np.asarray(
            [
                f"node_{i}_{j}"
                for i in range(len(self.rfs_))
                for j in range(self.rfs_[i].n_estimators)
            ],
            dtype=object,
        )

    def transform(self, X):
        check_is_fitted(self)
        _validate_data(
            self,
            X=X,
            reset=False,
            ensure_min_features=1,
            ensure_min_samples=1,
        )
        return np.hstack([rf.apply(X) for rf in self.rfs_])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.transformer_tags.preserves_dtype = []

        return tags
