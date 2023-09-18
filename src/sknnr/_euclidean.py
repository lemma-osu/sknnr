from sklearn.base import TransformerMixin

from ._base import TransformedKNeighborsRegressor
from .transformers import StandardScalerWithDOF


class EuclideanKNNRegressor(TransformedKNeighborsRegressor):
    def _get_transformer(self) -> TransformerMixin:
        return StandardScalerWithDOF(ddof=1)
