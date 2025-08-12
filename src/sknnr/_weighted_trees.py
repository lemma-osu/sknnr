from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._base import TransformedKNeighborsRegressor, YFitMixin
from .transformers import RFNodeTransformer

TREE_NODE_TRANSFORMER = RFNodeTransformer


class WeightedHammingDistanceMetric:
    """
    Distance metric to calculate Hamming distance based on weights.
    Setting of weights is only supported after initialization.
    """

    def __init__(self):
        self.w = None

    def set_weights(self, w: ArrayLike[float]):
        self.w = w

    def __call__(self, u: NDArray, v: NDArray) -> float:
        if self.w is None:
            self.w = np.ones_like(u)
        return np.sum((u != v) * self.w) / np.sum(self.w)


class WeightedTreesNNRegressor(YFitMixin, TransformedKNeighborsRegressor):
    """
    Base class for `TransformedKNeighbors` regressors that use a tree-based
    transformer and calculate distances using the Hamming metric.  Weights are set
    by the transformer based on forests (i.e. target attributes) and/or individual
    trees within each forest.  Weights on trees affect the Hamming distance
    calculation, where higher weights serve to accentuate dissimilarities between
    references and targets.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    weights : {"uniform", "distance"}, callable or None, default="uniform"
        Weight function used in prediction.
    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        Algorithm used to compute the nearest neighbors.
    leaf_size : int, default=30
        Leaf size passed to `BallTree` or `KDTree`.
    n_jobs : int, default=None
        The number of jobs to run in parallel.

    Notes
    -----
    Because this class does not implement the `_get_transformer` method, this class
    cannot be instantiated directly. Instead, use subclasses that implement the
    `_get_transformer` method.
    """

    transformer_: TREE_NODE_TRANSFORMER

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] | Callable = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        n_jobs: int | None = None,
    ):
        self.metric = WeightedHammingDistanceMetric()
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=self.metric,
            n_jobs=n_jobs,
        )

    def _set_fitted_transformer(self, X, y) -> None:
        super()._set_fitted_transformer(X, y)
        self.metric.set_weights(self.transformer_.tree_weights_)
