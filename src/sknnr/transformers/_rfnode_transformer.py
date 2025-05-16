from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_array, check_is_fitted

from .._base import _validate_data
from ..utils import get_feature_names_and_dtypes, is_nan_like, is_number_like_type

if TYPE_CHECKING:
    import pandas as pd


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

    See `sklearn.ensemble.RandomForestRegressor` for more detail on these
    parameters.  All parameters are passed through to `RandomForestRegressor`
    for each random forest being built.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of trees in _each_ random forest.
    criterion_reg : {"squared_error", "absolute_error", "friedman_mse", "poisson"},
        default="squared_error"
        The function to measure the quality of a split for RandomForestRegresor
        objects.
    criterion_clf : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split for RandomForestClassifier
        objects.
    max_depth : int, default=None
        The maximum depth of the tree.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int of float, default=5
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the
        input samples) required to be at a leaf node.
    max_features_reg : {“sqrt”, “log2”, None}, int or float, default=1.0
        The number of features to consider when looking for the best split for
        RandomForestRegressor objects.
    max_features_clf : {“sqrt”, “log2”, None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split for
        RandomForestClassifier objects.
    max_leaf_nodes : int, default=None
        Grow trees with max_leaf_nodes in best-first fashion.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
    n_jobs : int, default=None
        The number of jobs to run in parallel.
    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples
        used when building trees (if `bootstrap=True`) and the sampling of the
        features to consider when looking for the best split at each node
        (if `max_features < n_features`).
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    warm_start : bool, default=False
        When set to `True`, reuse the solution of the previous call to fit and
        add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    class_weight_clf : {“balanced”, “balanced_subsample”}, dict or list of dicts,
        default=None
        Weights associated with classes in the form {class_label: weight}. If not
        given, all classes are supposed to have weight one.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    max_samples : int or float, default=None
        If bootstrap is `True`, the number of samples to draw from X to
        train each base estimator.
    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during `fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`)
        Names of features seen during fit. Defined only when `X` has feature
        names that are all strings.
    rf_type_dict_ : dict[str, str]
        Dictionary mapping target names to their random forest type
        ("regression" or "classification").
    rfs_ : list of `RandomForestRegressor`
        The random forests associated with each target in `y` during `fit`.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        criterion_reg: Literal[
            "squared_error", "absolute_error", "friedman_mse", "poisson"
        ] = "squared_error",
        criterion_clf: Literal["gini", "entropy", "log_loss"] = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 5,
        min_weight_fraction_leaf: float = 0.0,
        max_features_reg: Literal["sqrt", "log2"] | int | float | None = 1.0,
        max_features_clf: Literal["sqrt", "log2"] | int | float | None = "sqrt",
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool | Callable = False,
        n_jobs: int | None = None,
        random_state: int | RandomState | None = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight_clf: Literal["balanced", "balanced_subsample"]
        | dict[str, float]
        | list[dict[str, float]]
        | None = None,
        ccp_alpha: float = 0.0,
        max_samples: int | float | None = None,
        monotonic_cst: list[int] | None = None,
    ):
        self.n_estimators = n_estimators
        self.criterion_reg = criterion_reg
        self.criterion_clf = criterion_clf
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features_reg = max_features_reg
        self.max_features_clf = max_features_clf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight_clf = class_weight_clf
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.monotonic_cst = monotonic_cst

    def _validate_and_promote_targets(
        self, y: Any, target_info: dict[str, np.dtype | pd.CategoricalDtype]
    ) -> list[NDArray]:
        """
        Given target names and types, validate and promote each target in `y`.

        `y` is treated as a 2D array, where each column is a target with potentially
        different dtypes between columns. Each target is first validated to have
        no NaN-like values and then promoted to the minimum numpy dtype that
        losslessly represents all elements (as previously captured in `target_info`).
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
                pass
                # target = np.asarray(target.tolist())

            # Check for targets with mixed numeric and non-numeric elements.
            # Non-lossy promotion of numeric types to other numeric types is
            # allowed (e.g. bool to int, int to float), but potentially lossy
            # promotion from numeric to non-numeric types is not allowed
            # (e.g. int to str, float to str).
            elif np.issubdtype(promoted_dtype, np.str_) and (
                non_string_types := {
                    type(v) for v in target if not np.issubdtype(type(v), np.str_)
                }
            ):
                raise ValueError(
                    f"Target {name} has non-string types ({non_string_types}) "
                    "that cannot be losslessly converted to a string dtype "
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

    def _set_rf_types(
        self, target_info: dict[str, Any]
    ) -> dict[str, Literal["regression", "classification"]]:
        """Set the random forest type to use for each target in `y`."""

        # TODO: Handle overrides from user based on names
        # TODO: target_info.update(user_overrides)
        return {
            k: "regression" if is_number_like_type(v) else "classification"
            for k, v in target_info.items()
        }

    def fit(self, X, y):
        _validate_data(self, X=X, reset=True)

        # Get target names and minimum numpy dtypes for each target in `y`
        target_info = get_feature_names_and_dtypes(y)

        # Validate and promote targets within `y`
        y = self._validate_and_promote_targets(y, target_info)

        # Assign RF types based on the target dtypes
        self.rf_type_dict_ = self._set_rf_types(target_info)

        # Specialize the kwargs sent to initialize the random forests
        rf_common_kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            monotonic_cst=self.monotonic_cst,
        )
        rf_reg_kwargs = {
            **rf_common_kwargs,
            "criterion": self.criterion_reg,
            "max_features": self.max_features_reg,
        }
        rf_clf_kwargs = {
            **rf_common_kwargs,
            "criterion": self.criterion_clf,
            "max_features": self.max_features_clf,
            "class_weight": self.class_weight_clf,
        }

        # Create the random forests for each target in `y` and fit them
        target_idx_to_rf_type = {
            i: v for i, (_, v) in enumerate(self.rf_type_dict_.items())
        }
        self.rfs_ = [
            RandomForestRegressor(**rf_reg_kwargs).fit(X, target)
            if target_idx_to_rf_type[i] == "regression"
            else RandomForestClassifier(**rf_clf_kwargs).fit(X, target)
            for i, target in enumerate(y)
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
