from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike

from ._base import TransformedKNeighborsRegressor, YFitMixin
from .transformers._tree_node_transformer import TreeNodeTransformer


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

    transformer_: TreeNodeTransformer
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
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric="hamming",
            n_jobs=n_jobs,
        )

    def _set_fitted_transformer(self, X, y) -> None:
        super()._set_fitted_transformer(X, y)
        self.hamming_weights_ = self._get_hamming_weights()

    def _get_hamming_weights(self):
        """
        Get the weights for the Hamming distance metric, based on tree weights
        from the transformer and forest weights provided as a user parameter.
        """
        # Equal weighting for all forests
        if isinstance(self.forest_weights, str) and self.forest_weights == "uniform":
            forest_weights_arr = np.ones(self.transformer_.n_forests_, dtype="float64")

        # User-supplied forest weights
        else:
            if len(self.forest_weights) != self.transformer_.n_forests_:
                raise ValueError(
                    "Expected `forest_weights` to have length "
                    f"{self.transformer_.n_forests_}, but got "
                    f"{len(self.forest_weights)}."
                )
            forest_weights_arr = np.asarray(self.forest_weights, dtype="float64")

        # Adjust forest weights based on whether the forest creates multiple trees
        # per iteration (i.e. classification with multiple classes in gradient
        # boosting).  This ensures that forests with more trees do not
        # disproportionately influence the distance metric.
        for i in range(len(forest_weights_arr)):
            forest_weights_arr[i] /= self.transformer_.n_trees_per_iteration_[i]

        # Element-wise multiply the transformer's tree weights with the forest_weights
        # and set the Hamming metric weights
        return np.hstack(
            [
                tw * fw
                for tw, fw in zip(self.transformer_.tree_weights_, forest_weights_arr)
            ]
        )

    def _get_additional_regressor_init_kwargs(self) -> dict:
        return {"metric_params": {"w": self.hamming_weights_}}
