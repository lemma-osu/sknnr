from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from ._tree_node_transformer import TreeNodeTransformer


class RFNodeTransformer(TreeNodeTransformer):
    """
    Transformer to capture node indexes for samples across multiple
    random forests.

    A random forest is fit to each `y` target in the training set using either
    scikit-learn's `RandomForestRegressor` or `RandomForestClassifier`.  The
    transformation captures the node indexes for each tree in each forest for
    each training or new sample.

    The particular random forest type used for each target is determined by the
    data type of the target.  If the target is numeric (e.g. `int` or `float`),
    a `RandomForestRegressor` is used.  If the target is categorical (e.g.
    `str` or `pd.Categorical`), a `RandomForestClassifier` is used.  Targets are
    automatically promoted to the minimum numpy dtype that safely represents
    all elements.

    This transformer is intended to be used in conjunction with `RFNNRegressor`
    which captures similarity between node indexes of training and inference data
    and creates predictions using nearest neighbors.

    See `sklearn.ensemble.RandomForestRegressor` and
    `sklearn.ensemble.RandomForestClassifier` for more detail on available
    parameters.  All parameters are passed through to these respective random
    forest estimators for each random forest being built.  Note that some
    parameters (e.g. `criterion` and `max_features`) are specified separately
    for regression and classification and have `_reg` and `_clf` suffixes.

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
    forest_weights: {"uniform"}, array-like of shape (n_targets), default="uniform"
        Weights assigned to each target in the training set when calculating
        Hamming distance between node indexes.  This allows for differential
        weighting of targets when calculating distances. Note that all trees
        associated with a target will receive the same weight.  If "uniform",
        each tree is assigned equal weight.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during `fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`)
        Names of features seen during fit. Defined only when `X` has feature
        names that are all strings.
    estimator_type_dict_ : dict[str, str]
        Dictionary mapping target names to their random forest type
        ("regression" or "classification").
    estimators_ : list [`RandomForestRegressor`|`RandomForestClassifier`]
        The random forests associated with each target in `y` during `fit`.
    tree_weights_ : ndarray of shape (n_trees,)
        Weights assigned to each tree in the forests to be used when calculating
        distances between node indexes.  If `forest_weights` is "uniform", each
        tree is assigned equal weight.  Otherwise, weights are assigned based
        on each target (forest) from `forest_weights` and all trees in that
        forest receive the same weight.
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
        forest_weights: Literal["uniform"] | ArrayLike[float] = "uniform",
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
        self.forest_weights = forest_weights

    def fit(self, X, y):
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
        self = self._fit(
            X,
            y,
            RandomForestRegressor,
            RandomForestClassifier,
            rf_reg_kwargs,
            rf_clf_kwargs,
        )

        if isinstance(self.forest_weights, str) and self.forest_weights == "uniform":
            # Assign equal weight to each tree
            self.tree_weights_ = np.ones(
                self.n_estimators * len(self.estimators_), dtype="float64"
            )
        else:
            # Assign weights by forest equally to all trees in that forest
            if len(self.forest_weights) != len(self.estimators_):
                raise ValueError(
                    f"Expected `forest_weights` to have length "
                    f"{len(self.estimators_)}, but got {len(self.forest_weights)}."
                )
            initial_weights = np.ones(
                (self.n_estimators, len(self.estimators_)), dtype="float64"
            )
            self.tree_weights_ = (self.forest_weights * initial_weights).T.flatten()
        return self

    def get_feature_names_out(self) -> NDArray:
        check_is_fitted(self, "estimators_")
        return np.asarray(
            [
                f"rf{i}_tree{j}"
                for i in range(len(self.estimators_))
                for j in range(self.estimators_[i].n_estimators)
            ],
            dtype=object,
        )
