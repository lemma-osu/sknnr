from sklearn.base import TransformerMixin

from ._base import TransformedKNeighborsRegressor
from .transformers import MahalanobisTransformer


class MahalanobisKNNRegressor(TransformedKNeighborsRegressor):
    """
    Nearest neighbor regression in an _n_-dimensional feature space where features
    have been transformed into Mahalanobis space using a `MahalanobisTransformer`,
    in order to facilitate simple Euclidean distance calculations in that space.
    See `MahalanobisTransformer` for more information on the transformation.

    See `sklearn.neighbors.KNeighborsRegressor` for more information on
    available parameters for k-neighbors regression used in instantiation.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for `kneighbors` queries.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors.
    leaf_size : int, default=30
        Leaf size passed to `BallTree` or `KDTree`.
    p : int, default=2
        Power parameter for the Minkowski metric.
    metric : str or callable, default='minkowski'
        The distance metric to use for the tree, calculated in standardized
        Euclidean space.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors search. `None` means
        1 unless in a `joblib.parallel_backend` context. `-1` means using all
        processors.

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
    transformer_ : MahalanobisTransformer
        The fitted transformer used to transform feature data.
    """

    def _get_transformer(self) -> TransformerMixin:
        return MahalanobisTransformer()
