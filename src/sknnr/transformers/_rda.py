import numpy as np

from ._constrained_ordination import ConstrainedOrdination


class RDA(ConstrainedOrdination):
    def _transform_X(self):
        self.env_center = np.average(self.X, axis=0)
        self.X = self.X - self.env_center

    def _transform_Y(self):
        self.Y = self.Y - self.Y.mean(axis=0)
        self.Y /= np.sqrt(self.Y.shape[0] - 1)
