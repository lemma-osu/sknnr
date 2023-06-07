from contextlib import contextmanager
from typing import Literal

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class StandardScalerWithDOF(StandardScaler):
    def __init__(self, ddof=0):
        super().__init__()
        self.ddof = ddof

    def fit(self, X, y=None, sample_weight=None):
        scaler = super().fit(X, y, sample_weight)
        scaler.scale_ = np.std(np.asarray(X), axis=0, ddof=self.ddof)
        return scaler


@contextmanager
def set_temp_output(
    transformer: TransformerMixin, temp_mode: Literal["default", "pandas"]
):
    """Temporarily set the output mode of an transformer."""
    previous_config = getattr(transformer, "_sklearn_output_config", {}).copy()

    transformer.set_output(transform=temp_mode)
    try:
        yield
    finally:
        transformer._sklearn_output_config = previous_config
