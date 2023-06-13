from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, TextIO, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

DATA_MODULE = "sknnr.datasets.data"


@dataclass
class Dataset:
    ids: Union[NDArray[np.int64], "pd.DataFrame"]
    data: Union[NDArray[np.float64], "pd.DataFrame"]
    target: Union[NDArray[np.float64], "pd.DataFrame"]
    frame: Union[None, "pd.DataFrame"]
    feature_names: list[str]
    target_names: list[str]

    def __repr__(self):
        n = self.data.shape[0]
        n_features = len(self.feature_names)
        n_targets = len(self.target_names)
        return f"Dataset(n={n}, features={n_features}, targets={n_targets})"


def _dataset_as_frame(dataset: Dataset) -> Dataset:
    """Convert a Dataset of arrays to a Dataset of DataFrames."""
    pd = _import_pandas()

    data_df = pd.DataFrame(dataset.data, columns=dataset.feature_names).set_index(
        dataset.ids
    )
    target_df = pd.DataFrame(dataset.target, columns=dataset.target_names).set_index(
        dataset.ids
    )

    frame = pd.concat([data_df, target_df], axis=1).set_index(dataset.ids)

    return Dataset(
        ids=dataset.ids,
        data=data_df,
        target=target_df,
        frame=frame,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
    )


def _open_text(module_name: str, file_name: str) -> TextIO:
    """Open a file as text.

    This is a compatibility port for importlib.resources.open_text, which is deprecated
    in Python>=3.9. This function will be removed when support for Python 3.8 is
    dropped.
    """
    if sys.version_info >= (3, 9):
        return resources.files(module_name).joinpath(file_name).open("r")
    return resources.open_text(module_name, file_name)


def _load_csv_data(
    file_name: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.unicode_]]:
    """Load data from a CSV file in the data module.

    Notes
    -----
    The CSV must be formatted with plot IDs in the first column and data values in the
    remaining columns. The first row must contain the column names.
    """
    with _open_text(DATA_MODULE, file_name) as csv_file:
        data_file = csv.reader(csv_file)
        headers = next(data_file)
        rows = list(iter(data_file))

        ids = np.array([row[0] for row in rows], dtype=np.int64)
        data = np.array([row[1:] for row in rows], dtype=np.float64)
        data_names = headers[1:]

    return ids, data, data_names


def load_moscow_stjoes(
    return_X_y: bool = False, as_frame: bool = False
) -> Union[tuple[NDArray[np.float64], NDArray[np.float64]], Dataset]:
    """Load the Moscow Mountain / St. Joe's dataset (Hudak 2010[^1]).

    The dataset contains 165 plots with environmental, LiDAR, and forest structure
    measurements. Structural measurements of basal area (BA) and tree density (TD)
    are separated by species.

    Parameters
    ----------
    return_X_y : bool, default=False
        If true, return the data and target as NumPy arrays instead of a Dataset.
    as_frame : bool, default=False
        If true, the `data` and `target` attributes of the returned Dataset will be
        DataFrames instead of NumPy arrays. The `frame` attribute will also added as a
        DataFrame with the `ids` as an index. Pandas must be installed for this option.

    Returns
    -------
    Dataset or tuple of ndarray
        A Dataset object containing the data, target, and feature names. If return_X_y
        is True, return a tuple of data and target arrays instead.

    Notes
    -----
    See Hudak 2010[^1] or https://cran.r-project.org/web/packages/yaImpute/yaImpute.pdf
    for more information on the dataset and feature names.

    Reference
    ---------
    [^1] Hudak, A.T. (2010) Field plot measures and predictive maps for "Nearest
    neighbor imputation of species-level, plot-scale forest structure attributes from
    LiDAR data". Fort Collins, CO: U.S. Department of Agriculture, Forest Service,
    Rocky Mountain Research Station.
    https://www.fs.usda.gov/rds/archive/Catalog/RDS-2010-0012
    """
    ids, data, feature_names = _load_csv_data(file_name="moscow_env.csv")
    _, target, target_names = _load_csv_data(file_name="moscow_spp.csv")

    moscow_stjoes = Dataset(
        ids=ids,
        data=data,
        target=target,
        feature_names=feature_names,
        target_names=target_names,
        frame=None,
    )

    if as_frame:
        moscow_stjoes = _dataset_as_frame(moscow_stjoes)

    return (moscow_stjoes.data, moscow_stjoes.target) if return_X_y else moscow_stjoes


def _import_pandas():
    """Import pandas and raise an error if it is not installed."""
    try:
        import pandas as pd
    except ImportError:
        msg = (
            "Pandas is required for this functionality. "
            "Please run `pip install pandas` and try again."
        )
        raise ImportError(msg) from None
    return pd
