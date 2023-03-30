from sklearn.pipeline import Pipeline

from ._base import IDNeighborsClassifier, KnnPipeline


class Raw(KnnPipeline):
    def _get_pipeline(self):
        classifier = IDNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
        )
        steps = [("classifier", classifier)]
        return Pipeline(steps)
