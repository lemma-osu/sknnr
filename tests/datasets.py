from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from sknnr.datasets import load_moscow_stjoes


@dataclass
class KNNTestDataset:
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.float64]
    y_test: NDArray[np.float64]
    ref_distances: NDArray[np.float64]
    ref_neighbors: NDArray[np.float64]
    trg_distances: NDArray[np.float64]
    trg_neighbors: NDArray[np.float64]
    trg_predicted_weighted: NDArray[np.float64]
    trg_predicted_unweighted: NDArray[np.float64]


def load_moscow_stjoes_results(method: str, k: int = 5) -> KNNTestDataset:
    """Load the Moscow Mountain / St. Joe's dataset results for port testing.

    The dataset will always be in dataframe format, with a fixed 80%/20% train/test
    split applied. New attributes are added with results from yaImpute using the given
    method and split.
    """
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=False
    )

    ref_distances = pd.read_csv(f"./tests/data/{method}_moscow_ref_distances_k{k}.csv")
    ref_neighbors = pd.read_csv(f"./tests/data/{method}_moscow_ref_neighbors_k{k}.csv")
    trg_distances = pd.read_csv(f"./tests/data/{method}_moscow_trg_distances_k{k}.csv")
    trg_neighbors = pd.read_csv(f"./tests/data/{method}_moscow_trg_neighbors_k{k}.csv")
    trg_predicted_weighted = pd.read_csv(
        f"./tests/data/{method}_moscow_trg_predicted_weighted_k{k}.csv"
    )
    trg_predicted_unweighted = pd.read_csv(
        f"./tests/data/{method}_moscow_trg_predicted_unweighted_k{k}.csv"
    )

    cols = [f"K{i+1}" for i in range(k)]

    return KNNTestDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ref_distances=ref_distances.loc[:, cols].values,
        ref_neighbors=ref_neighbors.loc[:, cols].values,
        trg_distances=trg_distances.loc[:, cols].values,
        trg_neighbors=trg_neighbors.loc[:, cols].values,
        trg_predicted_weighted=trg_predicted_weighted.iloc[:, 1:].values,
        trg_predicted_unweighted=trg_predicted_unweighted.iloc[:, 1:].values,
    )
