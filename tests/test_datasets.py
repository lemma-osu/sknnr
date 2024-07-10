import sys
from dataclasses import dataclass
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from sknnr.datasets import load_csv_data, load_moscow_stjoes, load_swo_ecoplot


@dataclass
class DatasetConfiguration:
    load_function: callable
    n_samples: int
    n_targets: int
    n_features: int


CONFIGURATIONS = {
    "moscow_stjoes": DatasetConfiguration(
        load_function=load_moscow_stjoes, n_samples=165, n_targets=35, n_features=28
    ),
    "swo_ecoplot": DatasetConfiguration(
        load_function=load_swo_ecoplot, n_samples=3005, n_targets=25, n_features=18
    ),
}


def test_passing_module_type():
    """Test that passing a module type to load_csv_data works."""
    import sknnr.datasets.data as data_module

    load_csv_data("moscow_env.csv", data_module=data_module)


def test_incorrect_module_raises_on_load_csv():
    """Test that load_csv_data raises when given an invalid module name."""
    invalid_module = "sknnr.datasets.invalid_module"
    with pytest.raises(ModuleNotFoundError, match="No module named"):
        load_csv_data("moscow_env.csv", data_module=invalid_module)


@pytest.mark.parametrize(
    "configuration", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys()
)
def test_load_dataset(configuration: DatasetConfiguration):
    """Test that the dataset is loaded with correct shapes and dtypes."""
    dataset = configuration.load_function()

    assert dataset.index.shape == (configuration.n_samples,)
    assert dataset.data.shape == (configuration.n_samples, configuration.n_features)
    assert dataset.target.shape == (configuration.n_samples, configuration.n_targets)
    assert len(dataset.feature_names) == configuration.n_features
    assert len(dataset.target_names) == configuration.n_targets
    assert dataset.frame is None

    assert dataset.index.dtype == np.int64
    assert dataset.data.dtype == np.float64
    assert dataset.target.dtype == np.float64
    assert isinstance(dataset.feature_names, list)
    assert isinstance(dataset.target_names, list)


@pytest.mark.parametrize(
    "configuration", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys()
)
def test_load_dataset_as_frame(configuration: DatasetConfiguration):
    """Test that the dataset is loaded as a dataframe."""
    dataset = configuration.load_function(as_frame=True)

    assert isinstance(dataset.frame, pd.DataFrame)
    assert isinstance(dataset.data, pd.DataFrame)
    assert isinstance(dataset.target, pd.DataFrame)

    assert dataset.data.shape == (configuration.n_samples, configuration.n_features)
    assert dataset.target.shape == (configuration.n_samples, configuration.n_targets)
    assert dataset.frame.shape == (
        configuration.n_samples,
        configuration.n_features + configuration.n_targets,
    )
    assert_array_equal(dataset.frame.index.values, dataset.index)


@pytest.mark.parametrize(
    "configuration", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys()
)
def test_load_dataset_as_xy(configuration: DatasetConfiguration):
    """Test that the dataset is loaded as X y arrays."""
    X, y = configuration.load_function(return_X_y=True)
    assert X.shape == (configuration.n_samples, configuration.n_features)
    assert y.shape == (configuration.n_samples, configuration.n_targets)


@pytest.mark.parametrize(
    "configuration", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys()
)
def test_load_dataset_as_xy_as_frame(configuration: DatasetConfiguration):
    """Test that the dataset is loaded as X y dataframes."""
    X, y = configuration.load_function(return_X_y=True, as_frame=True)
    data = configuration.load_function()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert X.shape == (configuration.n_samples, configuration.n_features)
    assert y.shape == (configuration.n_samples, configuration.n_targets)
    assert_array_equal(X.index.values, data.index)
    assert_array_equal(y.index.values, data.index)


@pytest.mark.parametrize(
    "configuration", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys()
)
def test_asframe_raises_without_pandas(configuration: DatasetConfiguration):
    """Test that as_frame=True raises a helpful error if pandas is not installed."""
    with mock.patch.dict(sys.modules, {"pandas": None}), pytest.raises(
        ImportError, match="pip install pandas"
    ):
        configuration.load_function(as_frame=True)


@pytest.mark.parametrize(
    "configuration", CONFIGURATIONS.values(), ids=CONFIGURATIONS.keys()
)
def test_dataset_repr(configuration: DatasetConfiguration):
    dataset = configuration.load_function()
    assert (
        repr(dataset)
        == f"Dataset(n={configuration.n_samples}, features={configuration.n_features},"
        f" targets={configuration.n_targets})"
    )
