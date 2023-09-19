from sklearn.base import TransformerMixin

from ._base import TransformedKNeighborsRegressor
from .transformers import MahalanobisTransformer


class MahalanobisKNNRegressor(TransformedKNeighborsRegressor):
    def _get_transformer(self) -> TransformerMixin:
        return MahalanobisTransformer()
