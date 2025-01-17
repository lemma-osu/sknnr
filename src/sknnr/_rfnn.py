import numbers

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri, r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ._base import (
    DFIndexCrosswalkMixin,
    IndependentPredictorMixin,
    YFitMixin,
    _validate_data,
)

# Functionality associated with R / rpy2
randomForest = importr("randomForest")


def rpy2_get_forest(X, y, n_tree, mt):
    """
    Train a random forest model in R using rpy2.
    """
    # Set seed in R for reproducibility
    ro.r("set.seed(42)")

    # # Train the random forest model in R
    with localconverter(pandas2ri.converter + numpy2ri.converter):
        xR = pandas2ri.py2rpy(X)
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
    with localconverter(pandas2ri.converter):
        xR = pandas2ri.py2rpy(X)
    nodes = r["attr"](r["predict"](rf, xR, proximity=False, nodes=True), "nodes")
    with localconverter(numpy2ri.converter):
        return numpy2ri.rpy2py(nodes)


class RFNNRegressor(
    DFIndexCrosswalkMixin, IndependentPredictorMixin, YFitMixin, BaseEstimator
):
    def __init__(
        self,
        n_neighbors=5,
        *,
        n_tree=500,
        n_estimators=100,
        mtry=None,
        weights="uniform",
    ):
        self.n_neighbors = n_neighbors
        self.n_tree = n_tree
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.weights = weights

    def fit(self, X, y):
        _validate_data(self, X=X, y=y, ensure_all_finite=True, multi_output=True)
        self._set_dataframe_index_in(X)

        self._fit_X = X
        self.rfs_ = []

        # Create a list of counts the same size as the number of columns in y
        # and populate with n_tree / num_columns with a minimum of 50
        n_tree_list = np.full(y.shape[1], max(50, self.n_tree // y.shape[1]))
        self.n_tree = n_tree_list.sum()

        # Set mtry
        mt = self.mtry if self.mtry else int(np.sqrt(X.shape[1]))

        # Build the individual random forests
        self.rfs_ = [
            rpy2_get_forest(X, y.values[:, i], int(n_tree_list[i]), mt)
            for i in range(y.shape[1])
        ]

        # Get the nodesets for each random forest
        self.nodesets_ = [rpy2_get_nodeset(rf, X) for rf in self.rfs_]

        return self

    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        return_dataframe_index=False,
    ):
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
            _validate_data(self, X=X, ensure_all_finite=True)

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
        if return_distance:
            neigh_dist = np.take_along_axis(all_matches, neigh_ind, axis=1)
            neigh_dist = (self.n_tree - neigh_dist) / self.n_tree

        if return_dataframe_index:
            msg = "Dataframe indexes can only be returned when fitted with a dataframe."
            check_is_fitted(self, "dataframe_index_in_", msg=msg)
            neigh_ind = self.dataframe_index_in_[neigh_ind]

        # Remove the sample itself from the neighbors
        if query_is_train:
            neigh_ind = neigh_ind[:, 1:]
            if return_distance:
                neigh_dist = neigh_dist[:, 1:]

        neigh_dist = neigh_dist[:, :n_neighbors]
        neigh_ind = neigh_ind[:, :n_neighbors]

        return neigh_dist, neigh_ind if return_distance else neigh_ind
