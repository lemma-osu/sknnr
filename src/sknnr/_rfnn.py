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
        n_estimators: int = 500,
        mtry: int | None = None,
        method: Literal["rpy2", "sklearn"] = "sklearn",
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
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.method = method

    def _get_transformer(self) -> TransformerMixin:
        return RFNodeTransformer(
            n_estimators=self.n_estimators, mtry=self.mtry, method=self.method
        )
