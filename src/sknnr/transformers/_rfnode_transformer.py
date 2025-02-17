from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .._base import _validate_data


class RFNodeTransformer(TransformerMixin, BaseEstimator):
    """
    Transformer to capture node indexes for samples across multiple
    random forests.

    A random forest is fit to each feature in the training set using
    `RandomForestRegressor`.  The transformation captures the node indexes
    for each tree in each forest for each training or target sample.

    This transformer is intended to be used in conjunction with `RFNNRegressor`
    which captures similarity between node indexes of training and target data
    and creates predictions using nearest neighbors.

    Parameters
    ----------
    n_estimators_per_forest : int, default=50
        Number of trees in each random forest.
    max_features : {'sqrt', 'log2', None}, int or float, default='sqrt'
        Number of features to consider when looking for the best split.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each forest.  See
        `RandomForestRegressor` for more information.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during `fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`)
        Names of features seen during fit. Defined only when `X` has feature
        names that are all strings.
    rfs_ : list of `RandomForestRegressor`
        The random forests associated with each feature in `y` during `fit`.
    """

    def __init__(
        self,
        n_estimators_per_forest: int = 50,
        max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
        random_state: int | RandomState | None = None,
    ):
        self.n_estimators_per_forest = n_estimators_per_forest
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        _, y = _validate_data(self, X=X, y=y, reset=True, multi_output=True)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.rfs_ = [
            RandomForestRegressor(
                n_estimators=self.n_estimators_per_forest,
                max_features=self.max_features,
                random_state=self.random_state,
                min_samples_leaf=5,
            ).fit(X, y[:, i])
            for i in range(y.shape[1])
        ]
        return self

    def get_feature_names_out(self) -> NDArray:
        check_is_fitted(self, "rfs_")
        return np.asarray(
            [
                f"rf{i}_tree{j}"
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

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.transformer_tags.preserves_dtype = ["int64"]

        return tags
