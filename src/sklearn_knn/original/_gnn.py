from sklearn.pipeline import Pipeline

from .._base import IDNeighborsClassifier, KnnPipeline
from ._cca_transformer import CCATransformer


class GNN(KnnPipeline):
    def _get_pipeline(self):
        transformer = CCATransformer()
        classifier = IDNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
        )
        steps = [("transform", transformer), ("classifier", classifier)]
        return Pipeline(steps)
