from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

from ._tree_node_transformer import TreeNodeTransformer


class GBNodeTransformer(TreeNodeTransformer):
    """
    Transformer to capture node indexes for samples across multiple
    gradient boosting estimators.

    A gradient boosting estimator is fit to each `y` target in the training set
    using either scikit-learn's `GradientBoostingRegressor` or
    `GradientBoostingClassifier`.  The transformation captures the node indexes
    for each tree in each estimator for each training or new sample.

    The particular gradient boosting estimator type used for each target is
    determined by the data type of the target.  If the target is numeric (e.g.
    `int` or `float`), a `GradientBoostingRegressor` is used.  If the target is
    categorical (e.g. `str` or `pd.Categorical`), a `GradientBoostingClassifier`
    is used.  Targets are automatically promoted to the minimum numpy dtype that
    safely represents all elements.

    This transformer is intended to be used in conjunction with `GBNNRegressor`
    which captures similarity between node indexes of training and inference
    data and creates predictions using nearest neighbors.

    See `sklearn.ensemble.GradientBoostingRegressor` and
    `sklearn.ensemble.GradientBoostingClassifier` for more detail on available
    parameters.  All parameters are passed through to these respective gradient
    boosting estimators for each model being built.  Note that some
    parameters (e.g. `loss` and `alpha`) are specified separately
    for regression and classification and have `_reg` and `_clf` suffixes.

    Parameters
    ----------
    loss_reg : {"squared_error", "absolute_error", "huber", "quantile"},
        default="squared_error"
        Loss function to be optimized for regression.
    loss_clf : {"log_loss", "exponential"}, default="log_loss"
        The loss function to be used for classification.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
    n_estimators : int, default=100
        The number of boosting stages to perform.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners.
    criterion : {"friedman_mse", "squared_error"}, default="friedman_mse"
        The function to measure the quality of a split.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the
        input samples) required to be at a leaf node.
    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    init : estimator, "zero" or None, default=None
        An estimator object that is used to compute the initial predictions.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each boosting
        iteration.
    max_features : {"sqrt", "log2"}, int or float, default=None
        The number of features to consider when looking for the best split.
    alpha_reg : float, default=0.9
        The alpha-quantile of the huber loss function and the quantile loss
        function.
    verbose : int, default=0
        Enable verbose output.
    max_leaf_nodes : int or None, default=None
        Grow trees with `max_leaf_nodes` in best-first fashion.
    warm_start : bool, default=False
        When set to `True`, reuse the solution of the previous call to fit and
        add more estimators to the ensemble, otherwise, just erase the previous
        solution.
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping.
    n_iter_no_change : int or None, default=None
        `n_iter_no_change` is used to decide if early stopping will be used to
        terminate training when validation score is not improving.
    tol : float, default=1e-4
        Tolerance for the early stopping.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during `fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`)
        Names of features seen during fit. Defined only when `X` has feature
        names that are all strings.
    estimator_type_dict_ : dict[str, str]
        Dictionary mapping target names to their gradient boosting type
        ("regression" or "classification").
    estimators_ : list [`GradientBoostingRegressor`|`GradientBoostingClassifier`]
        The gradient boosting models associated with each target in `y` during
        `fit`.
    n_forests_ : int
        The number of forests (i.e. targets) in the ensemble. Equal to
        `len(self.estimators_)`.
    tree_weights_ : ndarray of shape (n_forests_, n_estimators)
        Weights assigned to each tree in each forest to be used when calculating
        distances between node indexes.
    """

    def __init__(
        self,
        loss_reg: Literal[
            "squared_error", "absolute_error", "huber", "quantile"
        ] = "squared_error",
        loss_clf: Literal["log_loss", "exponential"] = "log_loss",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: Literal["friedman_mse", "squared_error"] = "friedman_mse",
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: int | None = 3,
        min_impurity_decrease: float = 0.0,
        init: BaseEstimator | Literal["zero"] | None = None,
        random_state: int | None = None,
        max_features: Literal["sqrt", "log2"] | int | float | None = None,
        alpha_reg: float = 0.9,
        verbose: int = 0,
        max_leaf_nodes: int | None = None,
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int | None = None,
        tol: float = 0.0001,
        ccp_alpha: float = 0.0,
    ):
        self.loss_reg = loss_reg
        self.loss_clf = loss_clf
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.alpha_reg = alpha_reg
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y):
        gb_common_kwargs = dict(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            init=self.init,
            random_state=self.random_state,
            max_features=self.max_features,
            verbose=self.verbose,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            ccp_alpha=self.ccp_alpha,
        )
        gb_reg_kwargs = {
            **gb_common_kwargs,
            "loss": self.loss_reg,
            "alpha": self.alpha_reg,
        }
        gb_clf_kwargs = {
            **gb_common_kwargs,
            "loss": self.loss_clf,
        }
        return self._fit(
            X,
            y,
            GradientBoostingRegressor,
            GradientBoostingClassifier,
            gb_reg_kwargs,
            gb_clf_kwargs,
        )

    def _set_tree_weights(self, X, y):
        tree_weights = []
        for est, target in zip(self.estimators_, y):
            # See https://github.com/lemma-osu/sknnr/issues/96#issuecomment-2967847215
            #
            # Calculate the initial loss, which by default in regression is based
            # on predicting the mean for all samples. The loss function is
            # half-squared error, so multiply by 2 to be consistent with how
            # sklearn calculates loss.
            initial_loss = (
                est._loss(np.asarray(target, dtype="float64"), est._raw_predict_init(X))
                * 2
            )

            # Calculate the change in loss at each stage
            loss_delta = np.diff(np.hstack([initial_loss, est.train_score_]))

            # Normalize the loss delta to get the relative contribution of each tree
            tree_weights.append(loss_delta / np.sum(loss_delta))

        return np.asarray(tree_weights, dtype="float64")

    def get_feature_names_out(self) -> NDArray:
        check_is_fitted(self, "estimators_")
        return np.asarray(
            [
                f"gb{i}_tree{j}"
                for i in range(len(self.estimators_))
                for j in range(self.estimators_[i].n_estimators)
            ],
            dtype=object,
        )
