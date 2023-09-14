import numpy as np
from numpy.typing import NDArray
from sklearn.base import TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted


class DFIndexCrosswalkMixin:
    """Mixin to crosswalk array indices to dataframe indexes."""

    def _set_dataframe_index_in(self, X):
        """Store dataframe indexes if X is a dataframe."""
        if hasattr(X, "index"):
            self.dataframe_index_in_ = np.asarray(X.index)


class IndependentPredictorMixin:
    """Mixin to return independent predictions based on the X data used
    for fitting the model."""

    def _set_independent_prediction_attributes(self, y):
        """Store independent predictions and score."""
        self.independent_prediction_ = super().predict(X=None)
        self.independent_score_ = super().score(X=None, y=y)


class _KNeighborsRegressor(
    DFIndexCrosswalkMixin, IndependentPredictorMixin, KNeighborsRegressor
):
    """
    Subclass of `sklearn.neighbors.KNeighborsRegressor` to support independent
    prediction and scoring and crosswalk array indices to dataframe indexes.

    This class serves as a superclass for many estimators in this package, but
    should not be instantiated directly.
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


class _TransformedKNeighborsRegressor(_KNeighborsRegressor):
    """
    Subclass for KNeighbors regressors that apply transformations to the feature data.

    This class serves as a superclass for many estimators in this package, but
    should not be instantiated directly.
    """

    transform_: TransformerMixin

    @property
    def feature_names_in_(self):
        return self.transform_.feature_names_in_

    @property
    def n_features_in_(self):
        return self.transform_.n_features_in_

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

    def _apply_transform(self, X) -> NDArray:
        """Apply the stored transform to the input data."""
        check_is_fitted(self, "transform_")
        return self.transform_.transform(X)

    def fit(self, X, y):
        """Fit using transformed feature data."""
        self._set_dataframe_index_in(X)
        X_transformed = self._apply_transform(X)
        return super().fit(X_transformed, y)

    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        return_dataframe_index=False,
    ):
        """Return neighbor indices and distances using transformed feature data."""
        X_transformed = self._apply_transform(X) if X is not None else X
        return super().kneighbors(
            X=X_transformed,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
            return_dataframe_index=return_dataframe_index,
        )
