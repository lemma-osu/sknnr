from __future__ import annotations

from typing import Callable, Literal

from sklearn.base import TransformerMixin
from sklearn.metrics import DistanceMetric

from ._base import TransformedKNeighborsRegressor, YFitMixin
from .transformers import CCATransformer


class GNNRegressor(YFitMixin, TransformedKNeighborsRegressor):
    """
    Regression using Gradient Nearest Neighbor (GNN) imputation.

    The target is predicted by local interpolation of the targets associated with
    the nearest neighbors in the training set, with distances calculated in transformed
    Canonical Correspondence Analysis (CCA) space.

    See `sklearn.neighbors.KNeighborsRegressor` for more information on parameters
    and implementation.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    n_components : int, optional
        Number of components to keep during CCA transformation. If `None`, all
        components are kept. If `n_components` is greater than the number of available
        components, an error will be raised.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors.
    leaf_size : int, default=30
        Leaf size passed to `BallTree` or `KDTree`.
    p : int, default=2
        Power parameter for the Minkowski metric.
    metric : str or callable, default='minkowski'
        The distance metric to use for the tree, calculated in CCA space.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors search. `None` means 1 unless
        in a `joblib.parallel_backend` context. `-1` means using all processors.

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric to use. It will be same as the `metric` parameter or a
        synonym of it, e.g. 'euclidean' if the `metric` parameter set to 'minkowski' and
        `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics will be
        same with `metric_params` parameter, but may also contain the `p` parameter
        value if the `effective_metric_` attribute is set to 'minkowski'.
    n_features_in_ : int
        Number of features seen during `fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature names
        that are all strings.
    n_samples_fit_ : int
        Number of samples in the fitted data.
    transformer_ : CCATransformer
        Fitted transformer.

    References
    ----------
    Ohmann JL, Gregory MJ. 2002. Predictive Mapping of Forest Composition and Structure
    with Direct Gradient Analysis and Nearest Neighbor Imputation in Coastal Oregon,
    USA. Canadian Journal of Forest Research, 32, 725–741.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        n_components: int | None = None,
        weights: Literal["uniform", "distance"] | Callable = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str | Callable | DistanceMetric = "minkowski",
        metric_params: dict | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.n_components = n_components

    def _get_transformer(self) -> TransformerMixin:
        return CCATransformer(self.n_components)

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
                "check_regressor_multioutput": "Row sums must be greater than 0.",
            },
        }
