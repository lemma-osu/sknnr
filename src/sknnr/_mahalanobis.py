from ._base import TransformedKNeighborsRegressor
from .transformers import MahalanobisTransformer


class MahalanobisKNNRegressor(TransformedKNeighborsRegressor):
    def _get_transformer(self):
        return MahalanobisTransformer()
