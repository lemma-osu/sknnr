import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from sknnr.datasets import load_moscow_stjoes


def test_load_moscow_stjoes():
    """Test that the dataset is loaded with correct shapes and dtypes."""
    moscow = load_moscow_stjoes()

    assert moscow.index.shape == (165,)
    assert moscow.data.shape == (165, 28)
    assert moscow.target.shape == (165, 35)
    assert len(moscow.feature_names) == 28
    assert len(moscow.target_names) == 35
    assert moscow.frame is None

    assert moscow.index.dtype == np.int64
    assert moscow.data.dtype == np.float64
    assert moscow.target.dtype == np.float64
    assert isinstance(moscow.feature_names, list)
    assert isinstance(moscow.target_names, list)


def test_load_moscow_stjoes_as_frame():
    """Test that the dataset is loaded as a dataframe."""
    moscow = load_moscow_stjoes(as_frame=True)

    assert isinstance(moscow.frame, pd.DataFrame)
    assert isinstance(moscow.data, pd.DataFrame)
    assert isinstance(moscow.target, pd.DataFrame)

    assert moscow.data.shape == (165, 28)
    assert moscow.target.shape == (165, 35)
    assert moscow.frame.shape == (165, 28 + 35)
    assert_array_equal(moscow.frame.index.values, moscow.index)


def test_load_moscow_stjoes_as_xy():
    """Test that the dataset is loaded as X y arrays."""
    X, y = load_moscow_stjoes(return_X_y=True)
    assert X.shape == (165, 28)
    assert y.shape == (165, 35)


def test_load_moscow_stjoes_as_xy_as_frame():
    """Test that the dataset is loaded as X y dataframes."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    data = load_moscow_stjoes()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert X.shape == (165, 28)
    assert y.shape == (165, 35)
    assert_array_equal(X.index.values, data.index)
    assert_array_equal(y.index.values, data.index)


def test_asframe_raises_without_pandas():
    """Test that as_frame=True raises a helpful error if pandas is not installed."""
    with mock.patch.dict(sys.modules, {"pandas": None}):
        with pytest.raises(ImportError, match="pip install pandas"):
            load_moscow_stjoes(as_frame=True)


def test_dataset_repr():
    moscow = load_moscow_stjoes()
    assert repr(moscow) == "Dataset(n=165, features=28, targets=35)"
