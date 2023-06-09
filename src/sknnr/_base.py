import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted


class KNeighborsDFIndexCrosswalkMixin:
    """
    Mixin for KNeighbors estimators that crosswalk array indices to dataframe indexes.
    """

    def fit(self, X, y):
        """Store dataframe indexes if X is a dataframe."""
        if hasattr(X, "index"):
            self.dataframe_index_in_ = np.asarray(X.index)

        return super().fit(X, y)

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


class TransformedKNeighborsMixin:
    """
    Mixin for KNeighbors regressors that apply transformations to the feature data.
    """

    transform_: TransformerMixin

    @property
    def feature_names_in_(self):
        return self.transform_.feature_names_in_

    def _check_feature_names(self, X, *, reset):
        """Override BaseEstimator._check_feature_names to prevent errors.

        This check would fail during fitting because `feature_names_in_` stores original
        names while X contains transformed names. We instead rely on the transformer to
        check feature names and warn or raise for mismatches.
        """
        return

    def _apply_transform(self, X) -> np.ndarray:
        """Apply the stored transform to the input data."""
        check_is_fitted(self, "transform_")
        self.transform_._validate_data(X, reset=False)
        return self.transform_.transform(X)

    def fit(self, X, y):
        """Fit using transformed feature data."""
        X_transformed = self._apply_transform(X)
        return super().fit(X_transformed, y)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Return neighbor indices and distances using transformed feature data."""
        X_transformed = self._apply_transform(X) if X is not None else X
        return super().kneighbors(
            X=X_transformed, n_neighbors=n_neighbors, return_distance=return_distance
        )
