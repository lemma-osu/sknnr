from ._base import TransformedKNeighborsRegressor, YFitMixin
from .transformers import CCorATransformer


class MSNRegressor(YFitMixin, TransformedKNeighborsRegressor):
    def __init__(self, n_components=None, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

    def _get_transformer(self):
        return CCorATransformer(self.n_components)

    def _more_tags(self):
        return {
            "multioutput": True,
            "requires_y": True,
        }
