import glob
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal

REGRESSION_DATA_DIR = Path("./tests/test_port")
YAIMPUTE_DATA_DIR = Path("./tests/data")

# Predict tests
for filename in glob.glob(str(REGRESSION_DATA_DIR / "test_predict*.npz")):
    # Get the regression data file name
    base_name = Path(filename).stem

    # Extract the parameters from the file name
    reference, weighted, full, method = base_name.split("_")[3:7]

    # Get the corresponding file in the yaimpute data directory
    reference = "ref" if reference == "reference" else "trg"
    full = "" if full == "full" else "_c3"
    stem = f"{method}_moscow_{reference}_predicted_{weighted}_k5{full}.csv"
    yaimpute_filename = YAIMPUTE_DATA_DIR / stem

    # Open both and compare the contents
    with np.load(filename) as data:
        regression_data = data["pred"]

    yaimpute_data = pd.read_csv(yaimpute_filename, index_col=0).values
    assert_array_almost_equal(regression_data, yaimpute_data, decimal=3)

# KNeighbors tests
for filename in glob.glob(str(REGRESSION_DATA_DIR / "test_kneighbors*.npz")):
    # Get the regression data file name
    base_name = Path(filename).stem

    # Extract the parameters from the file name
    reference, full, method = base_name.split("_")[3:6]

    # Get the corresponding file in the yaimpute data directory
    reference = "ref" if reference == "reference" else "trg"
    full = "" if full == "full" else "_c3"
    dist_stem = f"{method}_moscow_{reference}_distances_k5{full}.csv"
    nn_stem = f"{method}_moscow_{reference}_neighbors_k5{full}.csv"
    yaimpute_dist_filename = YAIMPUTE_DATA_DIR / dist_stem
    yaimpute_nn_filename = YAIMPUTE_DATA_DIR / nn_stem

    # Open both and compare the contents
    with np.load(filename) as data:
        regression_dist_data = data["dist"]
        regression_nn_data = data["nn"]

    yaimpute_dist_data = pd.read_csv(yaimpute_dist_filename, index_col=0).values
    assert_array_almost_equal(regression_dist_data, yaimpute_dist_data, decimal=3)

    yaimpute_nn_data = pd.read_csv(yaimpute_nn_filename, index_col=0).values
    assert_array_equal(regression_nn_data, yaimpute_nn_data)
