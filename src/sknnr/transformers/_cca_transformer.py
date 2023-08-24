import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from . import ComponentReducerMixin
from ._cca import CCA


class CCATransformer(ComponentReducerMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        X = self._validate_data(
            X,
            reset=True,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
            ensure_min_features=2,
            ensure_min_samples=1,
        )

        if not hasattr(y, "shape") or len(y.shape) < 2:
            raise ValueError("`y` must be a 2D array.")

        y = np.asarray(y)
        self.ordination_ = CCA(X, y)
        self.set_n_components()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = self._validate_data(
            X,
            reset=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=True,
            ensure_min_features=2,
            ensure_min_samples=1,
        )

        return (X - self.ordination_.env_center) @ self.ordination_.projector(
            n_components=self.n_components_
        )

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        unsupported_1d = "CCA requires 2D y arrays."

        return {
            "allow_nan": False,
            "requires_fit": True,
            "requires_y": True,
            "_xfail_checks": {
                "check_estimators_dtypes": unsupported_1d,
                "check_dtype_object": unsupported_1d,
                "check_estimators_fit_returns_self": unsupported_1d,
                "check_pipeline_consistency": unsupported_1d,
                "check_estimators_overwrite_params": unsupported_1d,
                "check_fit_score_takes_y": unsupported_1d,
                "check_estimators_pickle": unsupported_1d,
                "check_transformer_data_not_an_array": unsupported_1d,
                "check_transformer_general": unsupported_1d,
                "check_transformer_preserve_dtypes": unsupported_1d,
                "check_methods_sample_order_invariance": unsupported_1d,
                "check_methods_subset_invariance": unsupported_1d,
                "check_dict_unchanged": unsupported_1d,
                "check_dont_overwrite_parameters": unsupported_1d,
                "check_fit_idempotent": unsupported_1d,
                "check_fit_check_is_fitted": unsupported_1d,
                "check_n_features_in": unsupported_1d,
                "check_fit2d_predict1d": unsupported_1d,
                "check_fit2d_1sample": unsupported_1d,
                "check_estimators_nan_inf": unsupported_1d,
                "check_requires_y_none": unsupported_1d,
            },
        }
