from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import _is_arraylike, check_is_fitted

if TYPE_CHECKING:
    from .transformers._base import ComponentReducerMixin


class DFIndexCrosswalkMixin:
    """Mixin to crosswalk array indices to dataframe indexes."""

    def _set_dataframe_index_in(self, X):
        """Store dataframe indexes if X is a dataframe."""
        index = getattr(X, "index", None)
        if _is_arraylike(index):
            self.dataframe_index_in_ = np.asarray(index)


class IndependentPredictorMixin:
    """Mixin to return independent predictions based on the X data used
    for fitting the model."""

    def _set_independent_prediction_attributes(self, y):
        """Store independent predictions and score."""
        self.independent_prediction_ = super().predict(X=None)
        self.independent_score_ = super().score(X=None, y=y)


class YFitMixin:
    """Mixin for transformed estimators that use an optional y_fit to fit their
    transformer."""

    def _set_fitted_transformer(self, X, y):
        """Fit and store the transformer, using stored y_fit data if available."""
        y_fit = self.y_fit_ if self.y_fit_ is not None else y
        self.transformer_ = self._get_transformer().fit(X, y_fit)

    def fit(self, X, y, y_fit=None):
        """Fit using transformed feature data. If y_fit is provided, it will be used
        to fit the transformer."""
        self.y_fit_ = y_fit
        return super().fit(X, y)


class RawKNNRegressor(
    DFIndexCrosswalkMixin, IndependentPredictorMixin, KNeighborsRegressor
):
    """
    Subclass of `sklearn.neighbors.KNeighborsRegressor` to support independent
    prediction and scoring and crosswalk array indices to dataframe indexes.
    """

    def fit(self, X, y):
        """Override fit to set attributes using mixins."""
        self._set_dataframe_index_in(X)
        self = super().fit(X, y)
        self._set_independent_prediction_attributes(y)
        return self

    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        return_dataframe_index=False,
    ):
        """Override kneighbors to optionally return dataframe indexes."""
        neigh_dist, neigh_ind = super().kneighbors(
            X=X, n_neighbors=n_neighbors, return_distance=True
        )

        if return_dataframe_index:
            msg = "Dataframe indexes can only be returned when fitted with a dataframe."
            check_is_fitted(self, "dataframe_index_in_", msg=msg)
            neigh_ind = self.dataframe_index_in_[neigh_ind]

        return (neigh_dist, neigh_ind) if return_distance else neigh_ind


class TransformedKNeighborsRegressor(RawKNNRegressor, ABC):
    """
    Subclass for KNeighbors regressors that apply transformations to the feature data.

    This class serves as a superclass for many estimators in this package, but
    should not be instantiated directly.
    """

    transformer_: TransformerMixin

    @abstractmethod
    def _get_transformer(self) -> TransformerMixin:
        """Return the transformer to use for fitting. Must be implemented by
        subclasses."""
        ...

    def _set_fitted_transformer(self, X, y):
        """Fit and store the transformer."""
        self.transformer_ = self._get_transformer().fit(X, y)

    @property
    def feature_names_in_(self):
        return self.transformer_.feature_names_in_

    @property
    def n_features_in_(self):
        return self.transformer_.n_features_in_

    def _check_feature_names(self, X, *, reset):
        """Override BaseEstimator._check_feature_names to prevent errors.

        This check would fail during fitting because `feature_names_in_` stores original
        names while X contains transformed names. We instead rely on the transformer to
        check feature names and warn or raise for mismatches.
        """
        return

    def _check_n_features(self, X, *, reset):
        """Override BaseEstimator._check_n_features to prevent errors.

        See _check_feature_names.
        """
        return

    def fit(self, X, y):
        """Fit using transformed feature data."""
        self._validate_data(X, y, force_all_finite=True, multi_output=True)
        self._set_dataframe_index_in(X)
        self._set_fitted_transformer(X, y)

        X_transformed = self.transformer_.transform(X)
        return super().fit(X_transformed, y)

    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        return_dataframe_index=False,
    ):
        """Return neighbor indices and distances using transformed feature data."""
        check_is_fitted(self, "transformer_")
        X_transformed = self.transformer_.transform(X) if X is not None else X
        return super().kneighbors(
            X=X_transformed,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
            return_dataframe_index=return_dataframe_index,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.multi_label = True

        return tags


class OrdinationKNeighborsRegressor(TransformedKNeighborsRegressor, ABC):
    """
    Subclass for transformed KNeighbors regressors that apply ordination with
    dimensionality reduction.
    """

    transformer_: ComponentReducerMixin

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        n_components: int | None = None,
        weights: Literal["uniform", "distance"] | Callable = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str | Callable | DistanceMetric = "minkowski",
        metric_params: dict | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.n_components = n_components
