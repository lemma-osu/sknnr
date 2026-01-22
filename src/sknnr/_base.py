from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import _is_arraylike, check_is_fitted

if TYPE_CHECKING:
    from .transformers._base import ComponentReducerMixin


def _validate_data(estimator, *, ensure_all_finite: bool = True, **kwargs):
    """
    Compatibility wrapper around sklearn's _validate_data function.

    scikit-learn >= 1.6.0 moved _validate_data from a method of BaseEstimator to a
    public utility function. This function wraps the utility function if available,
    and falls back to the method if not. `force_all_finite` was also renamed to
    `ensure_all_finite`.

    TODO: Remove when sklearn < 1.6.0 support is dropped.
    """
    try:
        from sklearn.utils.validation import validate_data
    except ImportError:
        return estimator._validate_data(force_all_finite=ensure_all_finite, **kwargs)

    return validate_data(estimator, ensure_all_finite=ensure_all_finite, **kwargs)


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

    See `sklearn.neighbors.KNeighborsRegressor` for more information on
    available parameters for k-neighbors regression used in instantiation.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors.
    leaf_size : int, default=30
        Leaf size passed to `BallTree` or `KDTree`.
    p : int, default=2
        Power parameter for the Minkowski metric.
    metric : str or callable, default='minkowski'
        The distance metric to use for the tree, calculated in standardized
        Euclidean space.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors search. `None` means
        1 unless in a `joblib.parallel_backend` context. `-1` means using all
        processors.

    Attributes
    ----------
    effective_metric_ : str
        The distance metric to use. It will be same as the metric parameter
        or a synonym of it, e.g. 'euclidean' if the metric parameter set to
        'minkowski' and `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.
    independent_prediction_ : array-like of shape (n_samples, n_outputs)
        The independent predictions for each sample in the training set,
        obtained by calculating `kneighbors` on the training data itself and
        calculating predictions based on those neighbors.
    independent_score_ : float
        The independent score (i.e. coefficient of determination or R²) for
        the model, obtained by calculating the average R² across all outputs.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_samples_fit_ : int
        Number of samples in the fitted data.
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

        # To resolve potential floating point sorting issues, scale and
        # discretize distances, such that very close distances are grouped
        # together.  Sort first by these scaled distances, then by neighbor
        # index, ensuring a stable sort order.
        EPS = 1e-10
        scaled_distances = np.floor(neigh_dist / EPS).astype(np.int64)
        sorted_indices = np.lexsort((neigh_ind, scaled_distances), axis=1)

        neigh_dist = np.take_along_axis(neigh_dist, sorted_indices, axis=1)
        neigh_ind = np.take_along_axis(neigh_ind, sorted_indices, axis=1)

        if return_dataframe_index:
            msg = "Dataframe indexes can only be returned when fitted with a dataframe."
            check_is_fitted(self, "dataframe_index_in_", msg=msg)
            neigh_ind = self.dataframe_index_in_[neigh_ind]

        return (neigh_dist, neigh_ind) if return_distance else neigh_ind


class TransformedKNeighborsRegressor(BaseEstimator, ABC):
    """
    Subclass for KNeighbors regressors that apply transformations to the feature data.

    This class serves as a superclass for many estimators in this package, but
    should not be instantiated directly.
    """

    transformer_: TransformerMixin
    regressor_: RawKNNRegressor

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: Literal["uniform", "distance"] | Callable = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str | Callable | DistanceMetric = "minkowski",
        metric_params: dict | None = None,
        n_jobs: int | None = None,
    ):
        # Store initialization parameters for the RawKNNRegressor, but do not
        # instantiate it yet.  It will be instantiated during `fit`, after the
        # transformer has been fitted.
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    @abstractmethod
    def _get_transformer(self) -> TransformerMixin:
        """Return the transformer to use for fitting. Must be implemented by
        subclasses."""
        ...

    def _set_fitted_transformer(self, X, y):
        """Fit and store the transformer."""
        self.transformer_ = self._get_transformer().fit(X, y)

    def _get_additional_regressor_init_kwargs(self) -> dict:
        """Get any additional keyword arguments for the KNeighbors regressor
        initialization. Subclasses can override to provide additional arguments.
        """
        return {}

    def _transform_X(self, X):
        """Transform feature data using the fitted transformer."""
        check_is_fitted(self, "transformer_")
        return self.transformer_.transform(X) if X is not None else X

    def fit(self, X, y):
        """Fit using transformed feature data."""
        _validate_data(self, X=X, y=y, ensure_all_finite=True, multi_output=True)

        # Set the fitted transformer and apply the transformation which serves
        # as input to the KNeighbors regressor.  If estimators derive any custom
        # parameters to pass to the regressor, they should be set as estimator
        # attributes during `_set_fitted_transformer`.
        self._set_fitted_transformer(X, y)

        X_transformed = self.transformer_.transform(X)

        # Initialize and fit the KNeighbors regressor using the transformed data.
        # Override any additional regressor init kwargs provided by subclasses.
        reg_init_kwargs = {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric,
            "metric_params": self.metric_params,
            "n_jobs": self.n_jobs,
        }
        reg_init_kwargs.update(self._get_additional_regressor_init_kwargs())
        self.regressor_ = RawKNNRegressor(**reg_init_kwargs)
        self.regressor_.fit(X_transformed, y)

        # `X_transformed` is guaranteed to be array-like here, so we can set
        # dataframe indexes from `X` in the regressor if applicable.
        self.regressor_._set_dataframe_index_in(X)

        # Set the number of features to be equal to that of the transformed
        # features
        self.n_features_in_ = self.regressor_.n_features_in_

        # Copy over mixin attributes from the regressor
        self.independent_prediction_ = self.regressor_.independent_prediction_
        self.independent_score_ = self.regressor_.independent_score_
        if hasattr(self.regressor_, "dataframe_index_in_"):
            self.dataframe_index_in_ = self.regressor_.dataframe_index_in_

        return self

    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        return_dataframe_index=False,
    ):
        """Return neighbor indices and distances using transformed feature data."""
        X_transformed = self._transform_X(X)
        return self.regressor_.kneighbors(
            X=X_transformed,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
            return_dataframe_index=return_dataframe_index,
        )

    def predict(self, X):
        X_transformed = self._transform_X(X)
        return self.regressor_.predict(X_transformed)

    def score(self, X, y=None):
        X_transformed = self._transform_X(X)
        return self.regressor_.score(X_transformed, y)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False

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
