from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

from numpy.typing import ArrayLike, DTypeLike

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from pandas import CategoricalDtype
    from pandas.api.extensions import ExtensionDtype

NumpyDTypeLike = DTypeLike
"""Any data type that can be used in a numpy array."""

AnyDTypeLike = Union[NumpyDTypeLike, "ExtensionDtype"]
"""Any Numpy or Pandas data type, including extension types."""

DataDTypeLike = Union[NumpyDTypeLike, "CategoricalDtype"]
"""Data types used for features or targets, i.e. Numpy and Pandas categorical dtypes."""


class DataFrameLike(Protocol):
    """A protocol for dataframe-like objects, such as pandas and polars DataFrames."""

    columns: Sequence[Hashable]
    dtypes: Sequence[AnyDTypeLike]


class SeriesLike(Protocol):
    """A protocol for series-like objects, such as pandas and polars Series."""

    name: Hashable | None
    dtype: AnyDTypeLike


DataLike = DataFrameLike | SeriesLike | ArrayLike
"""
Data structures that can be used as input to Scikit-learn estimators, including labeled
and unlabeled arrays.

This broadly follows the "array-like" definition from Scikit-learn:
https://scikit-learn.org/stable/glossary.html#term-array-like
"""
