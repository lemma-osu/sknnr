from sklearn.utils.validation import check_is_fitted

from ._base import MyStandardScaler, IDNeighborsClassifier


class Euclidean(IDNeighborsClassifier):
    def fit(self, X, y):
        self.transform_ = MyStandardScaler().fit(X, y)
        X = self.transform_.transform(X)
        return super().fit(X, y)

    def predict(self, X):
        X = self.transform_.transform(X)
        return super().predict(X)

    def kneighbor_ids(self, X=None, n_neighbors=None):
        if X is not None:
            check_is_fitted(self, "transform_")
            X = self.transform_.transform(X)
        return super().kneighbor_ids(X=X, n_neighbors=n_neighbors)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is not None:
            check_is_fitted(self, "transform_")
            X = self.transform_.transform(X)
        return super().kneighbors(
            X=X, n_neighbors=n_neighbors, return_distance=return_distance
        )
