from sklearn.base import TransformerMixin

from ._base import OrdinationKNeighborsRegressor, YFitMixin
from .transformers import CCATransformer


class GNNRegressor(YFitMixin, OrdinationKNeighborsRegressor):
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
    independent_prediction_ : array-like of shape (n_samples, n_outputs)
        The independent predictions for each sample in the training set,
        obtained by calculating `kneighbors` on the training data itself and
        calculating predictions based on those neighbors.
    independent_score_ : float
        The independent score (i.e. coefficient of determination or R²) for
        the model, obtained by calculating the average R² across all outputs.
    n_features_in_ : int
        Number of features seen during `fit`.
    regressor_ : RawKNNRegressor
        The underlying RawKNNRegressor instance.
    transformer_ : CCATransformer
        The fitted CCA transformer used to tranform feature data.
    y_fit_ : array-like of shape (n_samples, n_targets)
        The target matrix seen during fit.  Note that `y_fit_` is only used for
        fitting, whereas regression will be run on the `y` values passed to
        `fit`.

    References
    ----------
    Ohmann JL, Gregory MJ. 2002. Predictive Mapping of Forest Composition and Structure
    with Direct Gradient Analysis and Nearest Neighbor Imputation in Coastal Oregon,
    USA. Canadian Journal of Forest Research, 32, 725–741.
    """

    def _get_transformer(self) -> TransformerMixin:
        return CCATransformer(self.n_components)
