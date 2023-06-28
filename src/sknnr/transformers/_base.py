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


class ComponentReducerMixin:
    """
    Mixin for transformers that allow reduction of the number of components.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def set_components(self, ordination_obj):
        n_components = (
            self.n_components
            if self.n_components is not None
            else ordination_obj.max_components
        )
        if not 0 <= n_components <= ordination_obj.max_components:
            raise ValueError(
                f"n_components={n_components} must be between 0 and "
                f"{ordination_obj.max_components}"
            )
        self.n_components_ = n_components
