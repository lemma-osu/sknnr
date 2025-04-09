from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .._base import _validate_data

if TYPE_CHECKING:  # pragma: no cover
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
    n_estimators: int, default=50
        The number of trees in _each_ random forest.
    criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"},
        default="squared_error"
        The function to measure the quality of a split.
    max_depth : int, default=None
        The maximum depth of the tree.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int of float, default=5
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the
        input samples) required to be at a leaf node.
    max_features : {“sqrt”, “log2”, None}, int or float, default=1.0
        The number of features to consider when looking for the best split.
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
    rfs_ : list of `RandomForestRegressor`
        The random forests associated with each feature in `y` during `fit`.
    """

    # Dictionary to map dtypes to random forest types
    DTYPE_TO_RF_TYPE = {
        "int": "regression",
        "int8": "regression",
        "int16": "regression",
        "int32": "regression",
        "int64": "regression",
        "uint8": "regression",
        "uint16": "regression",
        "uint32": "regression",
        "uint64": "regression",
        "Int8": "regression",
        "Int16": "regression",
        "Int32": "regression",
        "Int64": "regression",
        "UInt8": "regression",
        "UInt16": "regression",
        "UInt32": "regression",
        "UInt64": "regression",
        "float": "regression",
        "float16": "regression",
        "float32": "regression",
        "float64": "regression",
        "Float32": "regression",
        "Float64": "regression",
        "category": "classification",
        "object": "classification",
        "str": "classification",
        "string": "classification",
    }

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

    def _set_rf_types(self, y: NDArray | pd.Series | pd.DataFrame) -> dict[str, str]:
        """Set the random forest type to use for each feature in `y`."""
        try:
            column_types = {k: v.name for k, v in y.dtypes.items()}
            # TODO: Handle overrides from user based on names
            column_name_to_idx = {k: i for i, k in enumerate(y.columns)}
            column_types = {column_name_to_idx[k]: v for k, v in column_types.items()}
        except AttributeError:
            try:
                column_types = {y.name: y.dtype.name}
                # TODO: Handle overrides from user based on name
                column_types = {0: y.dtype.name}
            except AttributeError:
                if y.dtype != "object":
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)
                    column_types = {k: y.dtype.name for k in range(y.shape[1])}
                else:
                    column_types = {
                        k: type(y[0, k]).__name__ for k in range(y.shape[1])
                    }
                # TODO: Handle overrides from user based on index

        missing_types = set(column_types.values()) - set(self.DTYPE_TO_RF_TYPE.keys())
        if missing_types:
            msg = f"Dtypes {missing_types} are not supported for use "
            msg += "with random forests"
            raise KeyError(msg)

        return {k: self.DTYPE_TO_RF_TYPE[v] for k, v in column_types.items()}

    def fit(self, X, y):
        self.rf_type_dict_ = self._set_rf_types(y)
        _, y = _validate_data(self, X=X, y=y, reset=True, multi_output=True)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

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

        self.rfs_ = [
            RandomForestRegressor(**rf_reg_kwargs).fit(X, y[:, i])
            if self.rf_type_dict_[i] == "regression"
            else RandomForestClassifier(**rf_clf_kwargs).fit(X, y[:, i])
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
