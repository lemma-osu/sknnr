from .._base import IDNeighborsClassifier
from ._cca_transformer import CCATransformer


class GNN(IDNeighborsClassifier):
    """Normalized KNN"""
    def fit(self, X, y, cca_params=None):
        X = CCATransformer().fit_transform(X, **cca_params)
        return super().fit(X, y)
    
    def predict(self, X, cca_params=None):
        X = CCATransformer().fit_transform(X, **cca_params)
        return super().predict(X)