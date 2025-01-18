from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .._base import _validate_data
from . import ComponentReducerMixin, StandardScalerWithDOF
from ._ccora import CCorA


class CCorATransformer(ComponentReducerMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        _, y = _validate_data(self, X=X, y=y, reset=True, multi_output=True)
        self.scaler_ = StandardScalerWithDOF(ddof=1).fit(X)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = StandardScalerWithDOF(ddof=1).fit_transform(y)

        self.ordination_ = CCorA(self.scaler_.transform(X), y)
        self.set_n_components()
        self.projector_ = self.ordination_.projector(n_components=self.n_components_)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        _validate_data(self, X=X, reset=False, ensure_all_finite=True)
        return self.scaler_.transform(X) @ self.projector_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True

        return tags
