from sklearn.neighbors import KNeighborsRegressor

from ._base import IndependentPredictionMixin, KNeighborsDFIndexCrosswalkMixin


class RawKNNRegressor(
    KNeighborsDFIndexCrosswalkMixin,
    IndependentPredictionMixin,
    KNeighborsRegressor,
):
    pass
