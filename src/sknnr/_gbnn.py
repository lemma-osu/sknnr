from __future__ import annotations

from typing import Callable, Literal

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin

from ._weighted_trees import WeightedTreesNNRegressor
from .transformers import GBNodeTransformer


class GBNNRegressor(WeightedTreesNNRegressor):
    """
    Regression using Gradient Boosting Nearest Neighbors (GBNN) imputation.

    New data is predicted by similarity of its node indexes to training
    set node indexes when run through multiple univariate gradient boosting
    models.  A gradient boosting model is fit to each target in the training
    set and node indexes are captured for each tree in each forest for each
    training sample.  Node indexes are then captured for inference data and
    distance is calculated as the dissimilarity between node indexes.

    Gradient boosting models are constructed using either scikit-learn's
    `GradientBoostingRegressor` or `GradientBoostingClassifier` classes based on
    the data type of each target (`y` or `y_fit`) in the training set.  If the
    target is numeric (e.g. `int` or `float`), a `GradientBoostingRegressor` is
    used.  If the target is categorical (e.g. `str` or `pd.Categorical`), a
    `GradientBoostingClassifier` is used.  The
    `sknnr.transformers.GBNodeTransformer` class is responsible for constructing
    the gradient boosting models and capturing the node indexes.

    See `sklearn.neighbors.KNeighborsRegressor` for more detail on
    parameters associated with nearest neighbors.  See
    `sklearn.ensemble.GradientBoostingRegressor` and
    `sklearn.ensemble.GradientBoostingClassifier` for more detail on parameters
    associated with gradient boosting.  Note that some parameters (e.g.
    `loss` and `alpha`) are specified separately for regression and
    classification and have `_reg` and `_clf` suffixes.

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
    forest_weights: {"uniform"}, array-like of shape (n_targets), default="uniform"
        Weights assigned to each target in the training set when calculating
        Hamming distance between node indexes.  This allows for differential
        weighting of targets when calculating distances. Note that all trees
        associated with a target will receive the same weight.  If "uniform",
        each tree is assigned equal weight.
    tree_weighting_method: {"delta_loss", "uniform"}, default="delta_loss"
        The method used to weight the trees in each gradient boosting model.
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    weights : {"uniform", "distance"}, callable or None, default="uniform"
        Weight function used in prediction.
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        Algorithm used to compute the nearest neighbors.
    leaf_size : int, default=30
        Leaf size passed to `BallTree` or `KDTree`.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel.

    Attributes
    ----------
    effective_metric_ : str
        Always set to 'hamming'.
    effective_metric_params_ : dict
        Always empty.
    hamming_weights_ : np.array
        When `fit`, provides the weights on each tree in each forest when
        calculating the Hamming distance.
    independent_prediction_ : np.array
        When `fit`, provides the prediction for training data not allowing
        self-assignment during neighbor search.
    independent_score_ : double
        When `fit`, the mean coefficient of determination of the independent
        prediction across all features.
    n_features_in_ : int
        Number of features that the transformer outputs.  This is equal to the
        number of features in `y` (or `y_fit`) * `n_estimators_per_forest`.
    n_samples_fit_ : int
        Number of samples in the fitted data.
    transformer_ : GBNodeTransformer
        The fitted transformer which holds the built gradient boosting models
        for each feature.
    y_fit_ : np.array or pd.DataFrame
        When `y_fit` is passed to `fit`, the data used to construct the
        individual gradient boosting models.  Note that all `y` data is used
        for prediction.
    """

    def __init__(
        self,
        *,
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
        forest_weights: Literal["uniform"] | ArrayLike[float] = "uniform",
        tree_weighting_method: Literal["delta_loss", "uniform"] = "delta_loss",
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] | Callable = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        n_jobs: int | None = None,
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
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.alpha_reg = alpha_reg
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.forest_weights = forest_weights
        self.tree_weighting_method = tree_weighting_method

        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )

    def _get_transformer(self) -> TransformerMixin:
        return GBNodeTransformer(
            loss_reg=self.loss_reg,
            loss_clf=self.loss_clf,
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
            alpha_reg=self.alpha_reg,
            verbose=self.verbose,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            ccp_alpha=self.ccp_alpha,
            tree_weighting_method=self.tree_weighting_method,
        )
