from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike as NDTypeLike
    from pandas.api.extensions import ExtensionDtype

DTypeLike = Union["NDTypeLike", "ExtensionDtype"]


class DataFrameLike(Protocol):
    """A protocol for dataframe-like objects, such as pandas and polars DataFrames."""

    columns: Sequence[str]
    dtypes: Sequence[DTypeLike]


class SeriesLike(Protocol):
    """A protocol for series-like objects, such as pandas and polars Series."""

    name: str
    dtype: DTypeLike
