from sklearn.pipeline import Pipeline

from ._base import MyStandardScaler, IDNeighborsClassifier, KnnPipeline


class Euclidean(KnnPipeline):
    def _get_pipeline(self):
        scaler = MyStandardScaler()
        classifier = IDNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
        )
        steps = [("scaler", scaler), ("classifier", classifier)]
        return Pipeline(steps)
