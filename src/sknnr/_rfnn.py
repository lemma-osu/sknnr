import numbers

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from sklearn.neighbors._base import _get_weights
from sklearn.utils.validation import check_is_fitted

from ._base import RawKNNRegressor, YFitMixin, _validate_data

# Functionality associated with R / rpy2
randomForest = importr("randomForest")


def rpy2_get_forest(X, y, n_tree, mt):
    """
    Train a random forest model in R using rpy2.
    """
    # Set seed in R for reproducibility
    ro.r("set.seed(42)")

    # # Train the random forest model in R
    with localconverter(numpy2ri.converter):
        xR = numpy2ri.py2rpy(X)
        yR = numpy2ri.py2rpy(y.astype(np.float64))

    return randomForest.randomForest(
        x=xR,
        y=yR,
        proximity=False,
        importance=True,
        ntree=n_tree,
        keep_forest=True,
        mtry=mt,
    )


def rpy2_get_nodeset(rf, X):
    """
    Get the nodes associated with X of the random forest model in R using rpy2.
    """
    with localconverter(numpy2ri.converter):
        xR = numpy2ri.py2rpy(X)
    nodes = r["attr"](r["predict"](rf, xR, proximity=False, nodes=True), "nodes")
    with localconverter(numpy2ri.converter):
        return numpy2ri.rpy2py(nodes)


class RFNNRegressor(YFitMixin, RawKNNRegressor):
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        n_tree=500,
        n_estimators=100,
        mtry=None,
    ):
        super().__init__(n_neighbors=n_neighbors, weights=weights)
        self.n_tree = n_tree
        self.n_estimators = n_estimators
        self.mtry = mtry

    def fit(self, X, y):
        self._set_dataframe_index_in(X)
        self._fit_X, self._y = _validate_data(
            self, X=X, y=y, ensure_all_finite=True, multi_output=True
        )
        self.rfs_ = []

        # Create a list of counts the same size as the number of columns in y
        # and populate with n_tree / num_columns with a minimum of 50
        n_tree_list = np.full(self._y.shape[1], max(50, self.n_tree // y.shape[1]))
        self.n_tree = n_tree_list.sum()

        # Set mtry
        mt = self.mtry if self.mtry else int(np.sqrt(X.shape[1]))

        # Build the individual random forests
        self.rfs_ = [
            rpy2_get_forest(self._fit_X, self._y[:, i], int(n_tree_list[i]), mt)
            for i in range(y.shape[1])
        ]

        # Get the nodesets for each random forest
        self.nodesets_ = [rpy2_get_nodeset(rf, self._fit_X) for rf in self.rfs_]

        self._set_independent_prediction_attributes(y)
        return self

    def _kneighbors(self, X, n_neighbors=None, return_distance=True):
        check_is_fitted(self)

        # Repeated from KNeighborsMixin.kneighbors
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError(
                "n_neighbors does not take %s value, enter integer value"
                % type(n_neighbors)
            )

        # Repeated from KNeighborsMixin.kneighbors
        query_is_train = X is None
        if query_is_train:
            X = self._fit_X
        else:
            X = _validate_data(self, X=X, ensure_all_finite=True)

        # Create the count of matches between each sample in X and the
        # reference nodesets at the forest level
        forest_matches = []
        for i, rf in enumerate(self.rfs_):
            prd_nodes = rpy2_get_nodeset(rf, X)
            forest_matches.append(
                (prd_nodes[:, np.newaxis, :] == self.nodesets_[i]).sum(axis=-1)
            )

        # Sum the matches across the forests
        all_matches = np.dstack(forest_matches).sum(axis=-1)

        # Sort by number of matches
        # We invert all_matches to be "nonbecause ties are broken by the position in
        # the array and we want the
        inv_all_matches = self.n_tree - all_matches
        neigh_ind = np.apply_along_axis(
            lambda x: np.argsort(x, stable=True), 1, inv_all_matches
        )
        neigh_dist = np.take_along_axis(all_matches, neigh_ind, axis=1)
        neigh_dist = (self.n_tree - neigh_dist) / self.n_tree

        # Remove the sample itself from the neighbors
        if query_is_train:
            neigh_ind = neigh_ind[:, 1:]
            neigh_dist = neigh_dist[:, 1:]

        neigh_dist = neigh_dist[:, :n_neighbors]
        neigh_ind = neigh_ind[:, :n_neighbors]

        return (neigh_dist, neigh_ind) if return_distance else neigh_ind

    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        return_dataframe_index=False,
    ):
        neigh_dist, neigh_ind = self._kneighbors(
            X=X, n_neighbors=n_neighbors, return_distance=True
        )

        if return_dataframe_index:
            msg = "Dataframe indexes can only be returned when fitted with a dataframe."
            check_is_fitted(self, "dataframe_index_in_", msg=msg)
            neigh_ind = self.dataframe_index_in_[neigh_ind]

        return (neigh_dist, neigh_ind) if return_distance else neigh_ind

    def predict(self, X):
        # Repeated (verbatim) from KNeighborsRegressor.predict
        # other than call to self._kneighbors
        if self.weights == "uniform":
            # In that case, we do not need the distances to perform
            # the weighting so we do not compute them.
            neigh_ind = self._kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((neigh_dist.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred
