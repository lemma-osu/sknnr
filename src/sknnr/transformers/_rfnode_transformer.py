from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .._base import _validate_data
from ._rf_rpy2 import rpy2_get_forest, rpy2_get_nodeset
from ._rf_sklearn import sklearn_get_forest, sklearn_get_nodeset


class RFNodeTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 500,
        mtry: int | None = None,
        method: Literal["rpy2", "sklearn"] = "sklearn",
    ):
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.method = method

    def fit(self, X, y=None):
        X = _validate_data(
            self,
            X=X,
            reset=True,
            ensure_min_features=1,
            ensure_min_samples=1,
        )
        y = np.asarray(y)
        if len(y.shape) < 2:
            raise ValueError("`y` must be a 2D array.")

        # Create a list of counts the same size as the number of columns in y
        # and populate with n_tree / num_columns with a minimum of 50
        n_tree_list = np.full(y.shape[1], max(50, self.n_estimators // y.shape[1]))
        self.n_tree = n_tree_list.sum()

        mt = self.mtry if self.mtry else int(np.sqrt(X.shape[1]))

        # Build the individual random forests based on method
        if self.method == "rpy2":
            self.rfs_ = [
                rpy2_get_forest(X, y[:, i], int(n_tree_list[i]), mt)
                for i in range(y.shape[1])
            ]
        elif self.method == "sklearn":
            self.rfs_ = [
                sklearn_get_forest(X, y[:, i], int(n_tree_list[i]), mt)
                for i in range(y.shape[1])
            ]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = _validate_data(
            self,
            X=X,
            reset=False,
            ensure_min_features=1,
            ensure_min_samples=1,
        )
        if self.method == "rpy2":
            nodes = [rpy2_get_nodeset(rf, X) for rf in self.rfs_]
        elif self.method == "sklearn":
            nodes = [sklearn_get_nodeset(rf, X) for rf in self.rfs_]
        return np.hstack(nodes)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
