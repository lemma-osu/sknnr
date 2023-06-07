import warnings

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import _get_feature_names, check_is_fitted

from .transformers._base import set_temp_output


class IDNeighborsRegressor(KNeighborsRegressor):
    """
    Placeholder class for implementing plot ID access.
    """


class TransformedKNeighborsMixin(KNeighborsRegressor):
    """
    Mixin for KNeighbors regressors that apply transformations to the feature data.
    """

    @property
    def feature_names_in_(self):
        return self.transform_.feature_names_in_

    def _check_feature_names(self, X, *, reset):
        """Override BaseEstimator._check_feature_names to prevent errors.

        REFERENCE: https://github.com/scikit-learn/scikit-learn/blob/849c2f10f56b908abd9abbbffca8941494cc0bb0/sklearn/base.py#L409  # noqa: E501
        This is identical to the sklearn implementation except that:

        1. The reset option is ignored to avoid setting feature names.
        2. The second half of the method that validates X feature names against the
              fitted feature names is removed. This is because we want estimators to
              return the feature names that were used to fit the transformer, not the
              feature names that were used to fit the estimator.
        """
        fitted_feature_names = getattr(self, "feature_names_in_", None)
        X_feature_names = _get_feature_names(X)

        if fitted_feature_names is None and X_feature_names is None:
            return

        if X_feature_names is not None and fitted_feature_names is None:
            warnings.warn(
                f"X has feature names, but {self.__class__.__name__} was fitted without"
                " feature names",
                stacklevel=2,
            )
            return

        if X_feature_names is None and fitted_feature_names is not None:
            warnings.warn(
                "X does not have valid feature names, but"
                f" {self.__class__.__name__} was fitted with feature names",
                stacklevel=2,
            )
            return

    def _apply_transform(self, X) -> np.ndarray:
        """Apply the stored transform to the input data."""
        check_is_fitted(self, "transform_")

        # Temporarily run the transformer in pandas mode for dataframe inputs to ensure
        # that features are passed through to subsequent steps.
        output_mode = "pandas" if hasattr(X, "iloc") else "default"
        with set_temp_output(self.transform_, temp_mode=output_mode):  # type: ignore
            X_transformed = self.transform_.transform(X)

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
