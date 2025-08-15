from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .._base import _validate_data
from ..utils import get_feature_names_and_dtypes, is_nan_like, is_number_like_type

if TYPE_CHECKING:
    import pandas as pd


class TreeNodeTransformer(TransformerMixin, BaseEstimator, ABC):
    forest_weights: Literal["uniform"] | ArrayLike[float]

    def _validate_and_promote_targets(
        self, y: Any, target_info: dict[str, np.dtype | pd.CategoricalDtype]
    ) -> list[NDArray]:
        """
        Given target names and types, validate and promote each target in `y`.

        `y` is treated as a 2D array, where each column is a target with potentially
        different dtypes between columns. Each target is first validated to have
        no NaN-like values and then promoted to the minimum numpy dtype that
        safely represents all elements (as previously captured in `target_info`).
        Additionally, each target is validated to ensure no combination of
        string-like and non-string-like elements is present.

        Return the targets as a list of numpy arrays.
        """
        y = np.asarray(y, dtype=object)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        v_is_nan_like = np.vectorize(is_nan_like)
        targets = []
        for i, (name, promoted_dtype) in enumerate(target_info.items()):
            target = y[:, i]

            # Perform strict validation of the target to identify NaN-like
            # elements (None, np.nan, pd.NA).  We cannot use sklearn `check_array`
            # with `ensure_all_finite=True`` and `dtype=None`, as None values
            # go undetected and pd.NA values raise an error.
            if np.any(v_is_nan_like(target)):
                raise ValueError(f"Target {name} has NaN-like elements.")

            # If the promoted dtype is categorical, promote the data to the
            # minimum numpy dtype.  Numpy does not support categorical dtypes,
            # but we need to retain the categorical dtype label to correctly route
            # the target to a random forest classifier.
            if str(promoted_dtype) == "category":
                target = np.asarray(target.tolist())

            # Check for targets with mixed numeric and non-numeric elements.
            # Safe promotion of numeric types to other numeric types is
            # allowed (e.g. bool to int, int to float), but potentially unsafe
            # promotion from numeric to non-numeric types is not allowed
            # (e.g. int to str, float to str).
            elif np.issubdtype(promoted_dtype, np.str_) and (
                non_string_types := {
                    type(v) for v in target if not np.issubdtype(type(v), np.str_)
                }
            ):
                raise ValueError(
                    f"Target {name} has non-string types ({non_string_types}) "
                    f"that cannot be safely converted to a string dtype "
                    f"({promoted_dtype})."
                )

            # Otherwise, promote the target to the minimum numpy dtype
            else:
                target = target.astype(promoted_dtype)

            # Check for any other issues with the target when paired with the
            # estimator.
            target = check_array(
                target,
                ensure_all_finite=True,
                dtype=None,
                ensure_2d=False,
                estimator=self,
            )
            targets.append(target)

        return targets

    def _set_estimator_types(
        self, target_info: dict[str, Any]
    ) -> dict[str, Literal["regression", "classification"]]:
        """Set the estimator type to use for each target in `y`."""

        # TODO: Handle overrides from user based on names
        # TODO: target_info.update(user_overrides)
        return {
            k: "regression" if is_number_like_type(v) else "classification"
            for k, v in target_info.items()
        }

    def _fit(self, X, y, regressor_cls, classifier_cls, reg_kwargs, clf_kwargs):
        _validate_data(self, X=X, reset=True)

        if y is None:
            msg = (
                f"{self.__class__.__name__} requires y to be passed, "
                "but the target y is None."
            )
            raise ValueError(msg)

        # Get target names and minimum numpy dtypes for each target in `y`
        target_info = get_feature_names_and_dtypes(y)

        # Validate and promote targets within `y`
        y = self._validate_and_promote_targets(y, target_info)

        # Assign estimator types based on the target dtypes
        self.estimator_type_dict_ = self._set_estimator_types(target_info)

        # Create the estimators for each target in `y` and fit them
        target_idx_to_estimator_type = {
            i: v for i, (_, v) in enumerate(self.estimator_type_dict_.items())
        }
        self.estimators_ = [
            regressor_cls(**reg_kwargs).fit(X, target)
            if target_idx_to_estimator_type[i] == "regression"
            else classifier_cls(**clf_kwargs).fit(X, target)
            for i, target in enumerate(y)
        ]
        self.n_total_trees_ = len(self.estimators_) * self.estimators_[0].n_estimators
        return self

    def _set_tree_weights_from_forests(self):
        """
        Set the weights for each tree in the suite of forests based on user-
        specified values or equal weighting. The `forest_weights` attribute
        is set in subclasses and defaults to "uniform", which sets equal weight
        for each forest.  These weights are then used in the Hamming distance
        calculation between node indexes.

        Note that this method _only_ sets the tree weights based on forest
        weights such that each tree in a forest will have identical weight.
        Some derived transformers may further modify these weights based on
        individual trees.
        """
        if isinstance(self.forest_weights, str) and self.forest_weights == "uniform":
            # Assign equal weight to each tree
            self.tree_weights_ = np.ones(self.n_total_trees_, dtype="float64")
        else:
            # Ensure that forest_weights matches the number of forests
            if len(self.forest_weights) != len(self.estimators_):
                raise ValueError(
                    f"Expected `forest_weights` to have length "
                    f"{len(self.estimators_)}, but got {len(self.forest_weights)}."
                )

            # Assign weights by forest equally to all trees in that forest
            initial_weights = np.ones(
                (self.n_estimators, len(self.estimators_)), dtype="float64"
            )
            self.tree_weights_ = (self.forest_weights * initial_weights).T.flatten()

    @abstractmethod
    def fit(self, X, y): ...

    def transform(self, X):
        check_is_fitted(self)
        _validate_data(
            self,
            X=X,
            reset=False,
            ensure_min_features=1,
            ensure_min_samples=1,
        )
        return np.hstack([est.apply(X) for est in self.estimators_]).astype("int64")

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.transformer_tags.preserves_dtype = ["int64"]

        return tags
