from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .._base import _validate_data
from ..utils import (
    get_feature_names_and_dtypes,
    is_categorical_dtype,
    is_nan_like,
    is_number_like_dtype,
    is_numpy_dtypelike,
)

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any, Literal, Self

    from numpy.typing import NDArray
    from sklearn.utils._tags import Tags

    from ..types import DataDTypeLike, DataLike, TreeClassifier, TreeRegressor


def uniform_weights(n_forests: int, n_estimators: int) -> list[NDArray[np.float64]]:
    """
    Calculate uniform weights for an ensemble of tree-based estimators.
    """
    return [
        np.full(n_estimators, 1.0 / n_estimators, dtype=np.float64)
        for _ in range(n_forests)
    ]


class TreeNodeTransformer(TransformerMixin, BaseEstimator, ABC):
    def _validate_and_promote_targets(
        self, y: DataLike, target_info: dict[Hashable, DataDTypeLike]
    ) -> list[NDArray[np.object_ | np.number]]:
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
            # the target to a tree-based classifier.
            if is_categorical_dtype(promoted_dtype):
                target = np.asarray(target.tolist())

            elif is_numpy_dtypelike(promoted_dtype):
                # Check for targets with mixed numeric and non-numeric elements.
                # Safe promotion of numeric types to other numeric types is
                # allowed (e.g. bool to int, int to float), but potentially unsafe
                # promotion from numeric to non-numeric types is not allowed
                # (e.g. int to str, float to str).
                if np.issubdtype(promoted_dtype, np.str_) and (
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
        self, target_info: dict[Hashable, DataDTypeLike]
    ) -> dict[Hashable, Literal["regression", "classification"]]:
        """Set the estimator type to use for each target in `y`."""

        # TODO: Handle overrides from user based on names
        # TODO: target_info.update(user_overrides)
        return {
            k: "regression" if is_number_like_dtype(v) else "classification"
            for k, v in target_info.items()
        }

    def _fit(
        self,
        X: DataLike,
        y: DataLike,
        regressor_cls: type[TreeRegressor],
        classifier_cls: type[TreeClassifier],
        reg_kwargs: dict[str, Any],
        clf_kwargs: dict[str, Any],
    ) -> Self:
        X_arr = _validate_data(self, X=X, reset=True)

        if y is None:
            msg = (
                f"{self.__class__.__name__} requires y to be passed, "
                "but the target y is None."
            )
            raise ValueError(msg)

        # Get target names and minimum numpy dtypes for each target in `y`
        target_info = get_feature_names_and_dtypes(y)

        # Validate and promote targets within `y`
        y_arr = self._validate_and_promote_targets(y, target_info)

        # Assign estimator types based on the target dtypes
        self.estimator_type_dict_ = self._set_estimator_types(target_info)

        # Create the estimators for each target in `y` and fit them
        target_idx_to_estimator_type = {
            i: v for i, (_, v) in enumerate(self.estimator_type_dict_.items())
        }
        self.estimators_ = [
            regressor_cls(**reg_kwargs).fit(X_arr, target)
            if target_idx_to_estimator_type[i] == "regression"
            else classifier_cls(**clf_kwargs).fit(X_arr, target)
            for i, target in enumerate(y_arr)
        ]
        self.n_forests_ = len(self.estimators_)
        self.n_trees_per_iteration_ = self._set_n_trees_per_iteration()
        self.tree_weights_ = self._set_tree_weights(X_arr, y_arr)
        return self

    @abstractmethod
    def _set_n_trees_per_iteration(self) -> list[int]: ...

    @abstractmethod
    def _set_tree_weights(
        self,
        X: NDArray,
        y: list[NDArray[np.object_ | np.number]],
    ) -> list[NDArray[np.float64]]: ...

    @abstractmethod
    def fit(self, X: DataLike, y: DataLike) -> Self: ...

    def transform(self, X: DataLike) -> NDArray[np.int64]:
        check_is_fitted(self)
        X_arr = _validate_data(
            self,
            X=X,
            reset=False,
            ensure_min_features=1,
            ensure_min_samples=1,
        )

        # Get the node IDs for each tree in each forest
        node_ids = []
        for est in self.estimators_:
            est_node_ids = est.apply(X_arr)
            # In the case of some multi-class estimators (e.g.
            # GradientBoostingClassifier), the output of `apply` is 3D (n_samples,
            # n_estimators, n_classes). First swap axes to get (n_samples,
            # n_classes, n_estimators), then flatten the last two dimensions
            # to ensure a 2D output.
            if est_node_ids.ndim == 3:
                est_node_ids = np.swapaxes(est_node_ids, 1, 2).reshape(
                    est_node_ids.shape[0], -1
                )
            node_ids.append(est_node_ids)
        return np.hstack(node_ids).astype("int64")

    def fit_transform(self, X: DataLike, y: DataLike) -> NDArray[np.int64]:
        return self.fit(X, y).transform(X)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.transformer_tags.preserves_dtype = ["int64"]

        return tags
