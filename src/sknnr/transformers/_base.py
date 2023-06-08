import numpy as np
from sklearn.preprocessing import StandardScaler


class StandardScalerWithDOF(StandardScaler):
    def __init__(self, ddof=0):
        super().__init__()
        self.ddof = ddof

    def fit(self, X, y=None, sample_weight=None):
        scaler = super().fit(X, y, sample_weight)
        scaler.scale_ = np.std(np.asarray(X), axis=0, ddof=self.ddof)
        return scaler
