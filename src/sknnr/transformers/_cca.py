import numpy as np

from ._constrained_ordination import ConstrainedOrdination


class CCA(ConstrainedOrdination):
    def _transform_X(self):
        self.env_center = np.average(self.X, axis=0, weights=self.rw)
        X_scale = self.X - self.env_center
        self.X = X_scale * np.sqrt(self.rw)[:, np.newaxis]

    def _transform_Y(self):
        normalized = self.Y / self.Y.sum()
        self.rw = normalized.sum(axis=1)
        self.cw = normalized.sum(axis=0)
        rc = np.outer(self.rw, self.cw)
        self.Y = (normalized - rc) / np.sqrt(rc)
