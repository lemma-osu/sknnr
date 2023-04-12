from ._base import MyStandardScaler, IDNeighborsClassifier


class Euclidean(IDNeighborsClassifier):
    def fit(self, X, y):
        X = MyStandardScaler().fit_transform(X)
        return super().fit(X, y)
    
    def predict(self, X):
        X = MyStandardScaler().fit_transform(X)
        return super().predict(X)