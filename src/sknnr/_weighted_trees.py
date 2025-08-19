from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import hamming

from ._base import TransformedKNeighborsRegressor, YFitMixin
from .transformers import RFNodeTransformer

TREE_NODE_TRANSFORMER = RFNodeTransformer


class WeightedHammingDistanceMetric:
    """
    Distance metric to calculate Hamming distance based on weights. This class
    works as a callable pairwise distance metric and uses uniform weights by
    default. Note that setting of weights is only supported after initialization
    by using the `set_weights` method.
    """

    def __init__(self):
        self.w = None

    def set_weights(self, w: ArrayLike[float]):
        self.w = w

    def __call__(self, u: NDArray, v: NDArray) -> float:
        weights = self.w if self.w is not None else np.ones_like(u)
        return hamming(u, v, w=weights)


class WeightedTreesNNRegressor(YFitMixin, TransformedKNeighborsRegressor):
    """
    Base class for `TransformedKNeighbors` regressors that use a tree-based
    transformer and calculate distances using the Hamming metric.  Weights are a
    combination of tree-based weights set on each forest's trees by the transformer
    and user-supplied weights based on forests (i.e. target attributes).  Weights
    affect the Hamming distance calculation, where higher weights serve to
    accentuate dissimilarities between references and targets.

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
    forest_weights: Literal["uniform"] | ArrayLike[float]

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
        self.hamming_weights_ = self._get_hamming_weights()
        self.metric.set_weights(self.hamming_weights_)

    def _get_hamming_weights(self):
        """
        Get the weights for the Hamming distance metric, based on tree weights
        from the transformer and forest weights provided as a user parameter.
        """
        # Equal weighting for all forests
        if isinstance(self.forest_weights, str) and self.forest_weights == "uniform":
            forest_weights_arr = np.ones(
                (self.transformer_.n_forests_, 1), dtype="float64"
            )

        # User-supplied forest weights
        else:
            if len(self.forest_weights) != self.transformer_.n_forests_:
                raise ValueError(
                    "Expected `forest_weights` to have length "
                    f"{self.transformer_.n_forests_}, but got "
                    f"{len(self.forest_weights)}."
                )
            forest_weights_arr = np.asarray(
                self.forest_weights, dtype="float64"
            ).reshape(-1, 1)

        # Element-wise multiply the transformer's tree weights with the forest_weights
        # and set the Hamming metric weights
        return (forest_weights_arr * self.transformer_.tree_weights_).flatten()
