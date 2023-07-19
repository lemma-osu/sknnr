from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import ComponentReducerMixin, StandardScalerWithDOF
from ._ccora import CCorA


class CCorATransformer(ComponentReducerMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        self._validate_data(X, reset=True)

        self.scaler_ = StandardScalerWithDOF(ddof=1).fit(X)
        y = StandardScalerWithDOF(ddof=1).fit_transform(y)
        self.ordination_ = CCorA(self.scaler_.transform(X), y)
        self.set_n_components()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.scaler_.transform(X) @ self.ordination_.projector(
            n_components=self.n_components_
        )

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
