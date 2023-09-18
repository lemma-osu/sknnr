from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from sknnr.datasets import load_moscow_stjoes
from sknnr.datasets._base import _open_text

TEST_DATA_MODULE = "tests.data"


@dataclass
class KNNTestDataset:
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.float64]
    y_test: NDArray[np.float64]
    ref_distances: NDArray[np.float64]
    ref_neighbors: NDArray[np.float64]
    ref_predicted_weighted: NDArray[np.float64]
    ref_predicted_unweighted: NDArray[np.float64]
    trg_distances: NDArray[np.float64]
    trg_neighbors: NDArray[np.float64]
    trg_predicted_weighted: NDArray[np.float64]
    trg_predicted_unweighted: NDArray[np.float64]


def load_moscow_stjoes_results(
    method: str, k: int = 5, n_components: int | None = None
) -> KNNTestDataset:
    """Load the Moscow Mountain / St. Joe's dataset results for port testing.

    The dataset will always be in dataframe format, with a fixed 80%/20% train/test
    split applied. New attributes are added with results from yaImpute using the given
    method and split.
    """
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=False
    )

    n_components_str = f"_c{n_components}" if n_components is not None else ""

    ref_distances = _load_test_data(
        f"{method}_moscow_ref_distances_k{k}{n_components_str}.csv"
    )
    ref_neighbors = _load_test_data(
        f"{method}_moscow_ref_neighbors_k{k}{n_components_str}.csv"
    )
    ref_predicted_weighted = _load_test_data(
        f"{method}_moscow_ref_predicted_weighted_k{k}{n_components_str}.csv"
    )
    ref_predicted_unweighted = _load_test_data(
        f"{method}_moscow_ref_predicted_unweighted_k{k}{n_components_str}.csv"
    )
    trg_distances = _load_test_data(
        f"{method}_moscow_trg_distances_k{k}{n_components_str}.csv"
    )
    trg_neighbors = _load_test_data(
        f"{method}_moscow_trg_neighbors_k{k}{n_components_str}.csv"
    )
    trg_predicted_weighted = _load_test_data(
        f"{method}_moscow_trg_predicted_weighted_k{k}{n_components_str}.csv"
    )
    trg_predicted_unweighted = _load_test_data(
        f"{method}_moscow_trg_predicted_unweighted_k{k}{n_components_str}.csv"
    )

    cols = [f"K{i+1}" for i in range(k)]

    return KNNTestDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ref_distances=ref_distances.loc[:, cols].values,
        ref_neighbors=ref_neighbors.loc[:, cols].values,
        ref_predicted_weighted=ref_predicted_weighted.iloc[:, 1:].values,
        ref_predicted_unweighted=ref_predicted_unweighted.iloc[:, 1:].values,
        trg_distances=trg_distances.loc[:, cols].values,
        trg_neighbors=trg_neighbors.loc[:, cols].values,
        trg_predicted_weighted=trg_predicted_weighted.iloc[:, 1:].values,
        trg_predicted_unweighted=trg_predicted_unweighted.iloc[:, 1:].values,
    )


def _load_test_data(filename: str) -> pd.DataFrame:
    """Load a test dataset from the tests/data directory."""
    with _open_text(TEST_DATA_MODULE, filename) as fh:
        return pd.read_csv(fh)
