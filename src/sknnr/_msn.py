from sklearn.base import TransformerMixin

from ._base import OrdinationKNeighborsRegressor, YFitMixin
from .transformers import CCorATransformer


class MSNRegressor(YFitMixin, OrdinationKNeighborsRegressor):
    """
    Regression using Most Similar Neighbor (MSN) imputation.

    The target is predicted by local interpolation of the targets associated with
    the nearest neighbors in the training set, with distances calculated in transformed
    Canonical Correlation Analysis (CCorA) space.

    See `sklearn.neighbors.KNeighborsRegressor` for more information on parameters
    and implementation.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    n_components : int, optional
        Number of components to keep during CCorA transformation. If `None`, all
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
        The distance metric to use for the tree, calculated in CCorA space.
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
    transformer_ : CCorATransformer
        Fitted transformer.

    References
    ----------
    Moeur M, Stage AR. 1995. Most Similar Neighbor: An Improved Sampling Inference
    Procedure for Natural Resources Planning. Forest Science, 41(2), 337–359.
    """

    def _get_transformer(self) -> TransformerMixin:
        return CCorATransformer(self.n_components)
