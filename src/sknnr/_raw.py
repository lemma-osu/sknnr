from sklearn.neighbors import KNeighborsRegressor

from ._base import KNeighborsDFIndexCrosswalkMixin


class RawKNNRegressor(KNeighborsDFIndexCrosswalkMixin, KNeighborsRegressor):
    pass
