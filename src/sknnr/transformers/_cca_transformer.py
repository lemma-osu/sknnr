from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from .._base import _validate_data
from . import ComponentReducerMixin
from ._cca import CCA

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray
    from sklearn.utils._tags import Tags

    from ..types import DataLike


class CCATransformer(ComponentReducerMixin, TransformerMixin, BaseEstimator):
    """
    Transformer that performs Canonical Correspondence Analysis (CCA)
    (ter Braak, 1986). CCA is a constrained ordination technique that finds
    linear combinations of features that best explain variation in the target
    variables.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep during transformation. If `None`, all
        components are kept. If `n_components` is greater than the number of
        available components, an error will be raised.

    Attributes
    ----------
    env_center_ : ndarray of shape (n_features,)
        The center of the environmental (feature) data used for standardizing
        features during transformation.
    n_components_ : int
        The number of components kept during transformation.
    n_features_in_ : int
        Number of features seen during `fit`.
    ordination_ : CCA
        The fitted CCA ordination object.
    projector_ : ndarray of shape (n_features, n_components_)
        The projector matrix used to transform data.

    References
    ----------
    ter Braak, C.J.F. 1986. Canonical correspondence analysis: a new
    eigenvector technique for multivariate direct gradient analysis. Ecology,
    67: 1167–1179.
    """

    def fit(self, X: DataLike, y: DataLike) -> Self:
        X_arr = _validate_data(
            self,
            X=X,
            reset=True,
            dtype=FLOAT_DTYPES,
            ensure_all_finite=True,
            ensure_min_features=2,
            ensure_min_samples=1,
        )
        y = np.asarray(y)
        if len(y.shape) < 2:
            raise ValueError("`y` must be a 2D array.")

        self.ordination_ = CCA(X_arr, y)
        self.set_n_components()
        self.env_center_ = self.ordination_.env_center
        self.projector_ = self.ordination_.projector(n_components=self.n_components_)
        return self

    def transform(self, X: DataLike, y: None = None) -> NDArray:
        check_is_fitted(self)
        X_arr = _validate_data(
            self,
            X=X,
            reset=False,
            dtype=FLOAT_DTYPES,
            ensure_all_finite=True,
            ensure_min_features=2,
            ensure_min_samples=1,
        )
        return (X_arr - self.env_center_) @ self.projector_

    def fit_transform(self, X: DataLike, y: DataLike) -> NDArray:
        return self.fit(X, y).transform(X)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.positive_only = True

        return tags
