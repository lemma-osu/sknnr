from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .._base import _validate_data
from . import StandardScalerWithDOF

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray
    from sklearn.utils._tags import Tags

    from ..types import DataLike


class MahalanobisTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """
    Transformer that first standardizes features to zero mean and unit variance
    using a modified `StandardScaler` that uses N-1 degrees of freedom, then
    applies a Mahalanobis transformation to decorrelate features.

    When used in conjunction with Euclidean distance metrics, this is
    equivalent to calculating
    [Mahalanobis distances](https://en.wikipedia.org/wiki/Mahalanobis_distance)
    between samples.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during `fit`.
    scaler_ : StandardScalerWithDOF
        The fitted scaler used to standardize feature data.
    transform_ : ndarray of shape (n_features, n_features)
        The Mahalanobis transformation matrix.
    """

    def fit(self, X: DataLike, y: None = None) -> Self:
        _validate_data(
            self, X=X, ensure_all_finite="allow-nan", reset=True, ensure_min_features=2
        )

        self.scaler_ = StandardScalerWithDOF(ddof=1).fit(X)
        covariance = np.cov(self.scaler_.transform(X), rowvar=False)
        self.transform_ = np.linalg.inv(np.linalg.cholesky(covariance).T)
        return self

    def transform(self, X: DataLike, y: None = None) -> NDArray[np.float64]:
        check_is_fitted(self)
        _validate_data(self, X=X, ensure_all_finite="allow-nan", reset=False)

        return self.scaler_.transform(X) @ self.transform_

    def fit_transform(self, X: DataLike, y: None = None) -> NDArray[np.float64]:
        return self.fit(X, y).transform(X)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True

        return tags
