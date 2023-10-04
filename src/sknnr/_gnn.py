from __future__ import annotations

from typing import Literal

from sklearn.base import TransformerMixin

from ._base import TransformedKNeighborsRegressor, YFitMixin
from .transformers import ConstrainedTransformer


class GNNRegressor(YFitMixin, TransformedKNeighborsRegressor):
    def __init__(
        self,
        method: Literal["cca", "rda"] = "cca",
        n_components: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.n_components = n_components

    def _get_transformer(self) -> TransformerMixin:
        return ConstrainedTransformer(
            method=self.method, n_components=self.n_components
        )

    def _more_tags(self):
        unsupported_1d = "Constrained ordination requires 2D y arrays."

        return {
            "allow_nan": False,
            "requires_fit": True,
            "requires_y": True,
            "multioutput": True,
            "_xfail_checks": {
                "check_estimators_dtypes": unsupported_1d,
                "check_dtype_object": unsupported_1d,
                "check_estimators_fit_returns_self": unsupported_1d,
                "check_pipeline_consistency": unsupported_1d,
                "check_estimators_overwrite_params": unsupported_1d,
                "check_fit_score_takes_y": unsupported_1d,
                "check_estimators_pickle": unsupported_1d,
                "check_regressors_train": unsupported_1d,
                "check_regressor_data_not_an_array": unsupported_1d,
                "check_regressors_no_decision_function": unsupported_1d,
                "check_supervised_y_2d": unsupported_1d,
                "check_regressors_int": unsupported_1d,
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
                "check_regressor_multioutput": "Row sums must be greater than 0.",
            },
        }
