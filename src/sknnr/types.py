from collections.abc import Sequence
from typing import Protocol, TypeAlias

from numpy.typing import DTypeLike as NDTypeLike
from pandas.api.extensions import ExtensionDtype

DTypeLike: TypeAlias = NDTypeLike | ExtensionDtype


class DataFrameLike(Protocol):
    """A protocol for dataframe-like objects, such as pandas and polars DataFrames."""

    columns: Sequence[str]
    dtypes: Sequence[DTypeLike]


class SeriesLike(Protocol):
    """A protocol for series-like objects, such as pandas and polars Series."""

    name: str
    dtype: DTypeLike
