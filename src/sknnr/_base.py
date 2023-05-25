import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted


class NamedFeatureArray(np.ndarray):
    """An array with a columns attribute indicating feature names.

    Storing a `columns` attribute allows this array to act like  a dataframe for the
    purpose of extracting feature names when passed to sklearn estimators.
    """

    def __new__(cls, array, columns=None):
        obj = np.asarray(array).view(cls)
        obj.columns = columns
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.columns = getattr(obj, "columns", None)


class IDNeighborsRegressor(KNeighborsRegressor):
    """
    Placeholder class for implementing plot ID access.
    """


class TransformedKNeighborsMixin(KNeighborsRegressor):
    """
    Mixin for KNeighbors regressors that apply transformations to the feature data.
    """

    def _apply_transform(self, X) -> NamedFeatureArray:
        """Apply the stored transform to the input data.

        Note
        ----
        Transforming will cast input data to numpy arrays. To preserve feature names
        in the case of dataframe inputs, this method will wrap the transformed array
        in a `NamedFeatureArray` with a `columns` attribute, allowing `sklearn` to
        parse and store feature names.
        """
        check_is_fitted(self, "transform_")
        X_transformed = self.transform_.transform(X)
        if hasattr(X, "columns"):
            X_transformed = NamedFeatureArray(X_transformed, columns=X.columns)

        return X_transformed

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
