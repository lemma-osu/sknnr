from __future__ import annotations

from typing import Callable, Literal

from sklearn.base import TransformerMixin

from ._base import TransformedKNeighborsRegressor, YFitMixin
from .transformers import RFNodeTransformer


class RFNNRegressor(YFitMixin, TransformedKNeighborsRegressor):
    transformer_: TransformerMixin

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        n_estimators_per_forest: int = 50,
        max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
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
        self.n_estimators_per_forest = n_estimators_per_forest
        self.max_features = max_features

    def _get_transformer(self) -> TransformerMixin:
        return RFNodeTransformer(
            n_estimators_per_forest=self.n_estimators_per_forest,
            max_features=self.max_features,
        )
