from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from sknnr.datasets import load_moscow_stjoes


@dataclass
class TrainTestDataset:
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.float64]
    y_test: NDArray[np.float64]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Remove tests marked with `uncollect_if` from parametrization.

    See https://github.com/pytest-dev/pytest/issues/3730#issuecomment-567142496
    for discussion of how to filter parametrized tests using a custom hook
    and marker."""
    removed = []
    kept = []
    for item in items:
        m = item.get_closest_marker("uncollect_if")
        if m:
            func = m.kwargs["func"]
            if func(**item.callspec.params):
                removed.append(item)
                continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


@pytest.fixture(scope="session")
def moscow_stjoes_test_data() -> TrainTestDataset:
    """Load the Moscow Mountain / St. Joe's dataset for regression testing.

    The dataset will always be in dataframe format, with a fixed 80%/20% train/test
    split applied.
    """
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=False
    )

    return TrainTestDataset(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
