from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from numpy.typing import DTypeLike as NDTypeLike
    from pandas.api.extensions import ExtensionDtype

DTypeLike = Union["NDTypeLike", "ExtensionDtype"]


class Indexed(Protocol):
    """A protocol for objects that have an index, such as dataframes and series."""

    index: Sequence[Hashable]


class DataFrameLike(Indexed, Protocol):
    """A protocol for dataframe-like objects, such as pandas and polars DataFrames."""

    columns: Sequence[Hashable]
    dtypes: Sequence[DTypeLike]


class SeriesLike(Indexed, Protocol):
    """A protocol for series-like objects, such as pandas and polars Series."""

    name: Hashable | None
    dtype: DTypeLike


DataLike = DataFrameLike | SeriesLike | ArrayLike
"""
Data structures that can be used as input to Scikit-learn estimators, including labeled
and unlabeled arrays.

This broadly follows the "array-like" definition from Scikit-learn:
https://scikit-learn.org/stable/glossary.html#term-array-like
"""
