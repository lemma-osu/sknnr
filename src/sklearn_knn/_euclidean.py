from sklearn.utils.validation import check_is_fitted

from ._base import IDNeighborsClassifier, MyStandardScaler, TransformedKNeighborsMixin


class Euclidean(IDNeighborsClassifier, TransformedKNeighborsMixin):
    def fit(self, X, y):
        self.transform_ = MyStandardScaler().fit(X)
        X = self.transform_.transform(X)
        return super().fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = self.transform_.transform(X)
        return super().predict(X)
