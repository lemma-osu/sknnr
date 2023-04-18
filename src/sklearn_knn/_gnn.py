from sklearn.utils.validation import check_is_fitted
from ._base import IDNeighborsClassifier, TransformedKNeighborsMixin
from .original._cca_transformer import CCATransformer


class GNN(IDNeighborsClassifier, TransformedKNeighborsMixin):
    def fit(self, X, y, spp=None):
        self.transform_ = CCATransformer().fit(X, y=y, spp=spp)
        X = self.transform_.transform(X)
        return super().fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = self.transform_.transform(X)
        return super().predict(X)
