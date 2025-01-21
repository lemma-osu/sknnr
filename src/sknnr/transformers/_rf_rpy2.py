import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

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
