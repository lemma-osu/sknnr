from ._base import _TransformedKNeighborsRegressor
from .transformers import CCATransformer


class GNNRegressor(_TransformedKNeighborsRegressor):
    def __init__(self, n_components=None, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

    def fit(self, X, y, y_fit=None):
        y_fit = y_fit if y_fit is not None else y
        self.transform_ = CCATransformer(self.n_components).fit(X, y=y_fit)
        return super().fit(X, y)

    def _more_tags(self):
        unsupported_1d = "CCA requires 2D y arrays."

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
                "check_requires_y_none": unsupported_1d,
                "check_regressor_multioutput": "Row sums must be greater than 0.",
            },
        }
