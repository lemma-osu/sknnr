from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from sknnr.datasets import load_moscow_stjoes


@dataclass
class KNNTestDataset:
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.float64]
    y_test: NDArray[np.float64]


def load_moscow_stjoes_test_data() -> KNNTestDataset:
    """Load the Moscow Mountain / St. Joe's dataset for regression testing.

    The dataset will always be in dataframe format, with a fixed 80%/20% train/test
    split applied.
    """
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=False
    )

    return KNNTestDataset(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
