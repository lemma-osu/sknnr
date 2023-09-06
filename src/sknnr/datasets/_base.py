from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, TextIO

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

DATA_MODULE = "sknnr.datasets.data"


@dataclass
class Dataset:
    index: NDArray[np.int64] | pd.DataFrame
    data: NDArray[np.float64] | pd.DataFrame
    target: NDArray[np.float64] | pd.DataFrame
    frame: None | pd.DataFrame
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
        dataset.index
    )
    target_df = pd.DataFrame(dataset.target, columns=dataset.target_names).set_index(
        dataset.index
    )

    frame = pd.concat([data_df, target_df], axis=1).set_index(dataset.index)

    return Dataset(
        index=dataset.index,
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

        index = np.array([row[0] for row in rows], dtype=np.int64)
        data = np.array([row[1:] for row in rows], dtype=np.float64)
        data_names = headers[1:]

    return index, data, data_names


def _load_dataset_from_csv_filenames(
    *,
    data_filename: str,
    target_filename: str,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | Dataset:
    """Load separate data and target CSV files into a dataset or paired NumPy arrays.

    Parameters
    ----------
    data_filename: str, required
        The filename of the data CSV file.
    target_filename: str, required
        The filename of the target CSV file.
    return_X_y : bool, default=False
        If True, return the data and target as NumPy arrays instead of a Dataset.
    as_frame : bool, default=False
        If True, the `data` and `target` attributes of the returned Dataset will be
        DataFrames instead of NumPy arrays. The `frame` attribute will also be added as
        a DataFrame with the dataset index. Pandas must be installed for this
        option.

    Returns
    -------
    Dataset or tuple of ndarray
        A Dataset object containing the data, target, and feature names. If return_X_y
        is True, return a tuple of data and target arrays instead.

    Notes
    -----
    Both CSV files must be formatted with plot IDs in the first column and data values
    in the remaining columns. The first row in each file must contain the column names.
    The plot IDs in each file are expected to match and be in the same order.
    """
    index, data, feature_names = _load_csv_data(file_name=data_filename)
    _, target, target_names = _load_csv_data(file_name=target_filename)

    dataset = Dataset(
        index=index,
        data=data,
        target=target,
        feature_names=feature_names,
        target_names=target_names,
        frame=None,
    )

    if as_frame:
        dataset = _dataset_as_frame(dataset)

    return (dataset.data, dataset.target) if return_X_y else dataset


def load_moscow_stjoes(
    return_X_y: bool = False, as_frame: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | Dataset:
    """Load the Moscow Mountain / St. Joe's dataset (Hudak 2010[^1]).

    The dataset contains 165 plots with environmental, LiDAR, and forest structure
    measurements. Structural measurements of basal area (BA) and tree density (TD)
    are separated by species.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, return the data and target as NumPy arrays instead of a Dataset.
    as_frame : bool, default=False
        If True, the `data` and `target` attributes of the returned Dataset will be
        DataFrames instead of NumPy arrays. The `frame` attribute will also be added as
        a DataFrame with the dataset index. Pandas must be installed for this
        option.

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
    return _load_dataset_from_csv_filenames(
        data_filename="moscow_env.csv",
        target_filename="moscow_spp.csv",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


def load_swo_ecoplot(
    return_X_y: bool = False, as_frame: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | Dataset:
    """Load the southwest Oregon (SWO) USFS Region 6 Ecoplot dataset.

    The dataset contains 3,005 plots with environmental, Landsat, and forest cover
    measurements. Ocular measurements of tree cover (COV) are categorized by
    major tree species present in southwest Oregon.  All data were collected in 2000
    and Landsat imagery processed through the CCDC algorithm was extracted for the
    same year.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, return the data and target as NumPy arrays instead of a Dataset.
    as_frame : bool, default=False
        If True, the `data` and `target` attributes of the returned Dataset will be
        DataFrames instead of NumPy arrays. The `frame` attribute will also be added as
        a DataFrame with the dataset index. Pandas must be installed for this
        option.

    Returns
    -------
    Dataset or tuple of ndarray
        A Dataset object containing the data, target, and feature names. If return_X_y
        is True, return a tuple of data and target arrays instead.

    Notes
    -----
    These data are a subset of the larger USDA Forest Service Region 6 Ecoplot
    database, which holds 28,000 plots on Region 6 National Forests across Oregon
    and Washington.  The larger database is managed by Patricia Hochhalter (USFS Region
    6 Ecology Program) and used by permission.  Ecoplots were originally used to
    develop plant association guides and are used for a wide array of applications.
    This subset represents plots that were collected in southwest Oregon in 2000.

    Reference
    ---------
    Atzet, T, DE White, LA McCrimmon, PA Martinez, PR Fong, and VD Randall. 1996.
    Field guide to the forested plant associations of southwestern Oregon.
    USDA Forest Service. Pacific Northwest Region, Technical Paper R6-NR-ECOL-TP-17-96.
    """
    return _load_dataset_from_csv_filenames(
        data_filename="swo_ecoplot_env.csv",
        target_filename="swo_ecoplot_spp.csv",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


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
