from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import TypeGuard

    from pandas import CategoricalDtype

    from ..types import (
        AnyDTypeLike,
        DataDTypeLike,
        DataFrameLike,
        DataLike,
        NumpyDTypeLike,
        SeriesLike,
    )


def is_dataframe_like(obj: object) -> TypeGuard[DataFrameLike]:
    """
    Check if `obj` is a dataframe-like structure.  This will evaluate to `True`
    for pandas and polars DataFrames without the need to import those packages
    explicitly.
    """
    return hasattr(obj, "columns") and hasattr(obj, "dtypes")


def is_series_like(obj: object) -> TypeGuard[SeriesLike]:
    """
    Check if `obj` is a series-like structure.  This will evaluate to `True`
    for pandas and polars Series without the need to import those packages
    explicitly.
    """
    return hasattr(obj, "name") and hasattr(obj, "dtype")


def is_number_like_dtype(t: AnyDTypeLike) -> bool:
    """
    Check if `t` is a number-like dtype.  For most dtypes, np.issubdtype will
    correctly identify the type.  For pandas extension dtypes, we can check the
    kind of the dtype.
    """
    if is_numpy_dtypelike(t):
        return np.issubdtype(t, np.number)
    if kind := getattr(t, "kind", None):
        return kind in "iuf"
    raise TypeError(f"Unsupported type {t}")


def is_nan_like(x: object) -> bool:
    """Check if `x` is NaN-like, including None, np.nan, and pd.NA values."""
    return bool(
        x is None
        or (isinstance(x, float) and np.isnan(x))
        or x.__class__.__name__ == "NAType"
    )


def is_numpy_dtypelike(t: AnyDTypeLike) -> TypeGuard[NumpyDTypeLike]:
    """
    Check if `t` is a valid Numpy dtype. This narrows the type by excluding Pandas
    extension types like categorical.
    """
    try:
        np.dtype(t)  # type: ignore[arg-type]
        return True
    except TypeError:
        return False


def is_categorical_dtype(t: AnyDTypeLike) -> TypeGuard[CategoricalDtype]:
    """
    Check if `t` is a categorical dtype using duck-typing.
    """
    return str(t) == "category"


def get_feature_names(obj: DataLike | None) -> list[Hashable]:
    """
    Get the names of the features in `obj`. If no names are found, return
    a list of strings with the feature index.
    """
    if obj is None or len(np.asarray(obj)) == 0:
        return []
    if is_dataframe_like(obj):
        return list(obj.columns)
    if is_series_like(obj):
        return ["0"] if obj.name is None else [obj.name]
    arr = np.asarray(obj, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return [str(i) for i in range(arr.shape[1])]


def get_minimum_dtypes(obj: DataLike) -> list[NumpyDTypeLike]:
    """
    Return the smallest numpy dtype that can accommodate all data for each
    column in obj.
    """
    arr = np.asarray(obj, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return [np.asarray(arr[:, i].tolist()).dtype for i in range(arr.shape[1])]


def get_feature_dtypes(obj: DataLike | None) -> Sequence[DataDTypeLike]:
    """
    Get numpy dtypes of the features in `obj`, promoting dtypes as necessary to
    the smallest type that can accommodate all data elements in that feature.
    The exception to this is when a pd.CategoricalDtype feature is present,
    retain this dtype.

    Parameters
    ----------
    obj : array-like, DataFrame, or Series, optional
        The object from which to get the feature dtypes.

    Returns:
    --------
    list of dtypes : list[DTypeLike]
        The dtypes (either numpy or pd.CategoricalDtype) of the features in `obj`.
    """
    if obj is None or len(np.asarray(obj)) == 0:
        return []

    if is_dataframe_like(obj):
        native_dtypes = getattr(obj.dtypes, "values", obj.dtypes)
        promoted_dtypes = get_minimum_dtypes(obj)
        return [
            n_dtype if is_categorical_dtype(n_dtype) else p_dtype
            for p_dtype, n_dtype in zip(promoted_dtypes, native_dtypes, strict=True)
        ]
    if is_series_like(obj):
        native_dtype = obj.dtype
        promoted_dtype = get_minimum_dtypes(obj)[0]
        return [native_dtype if is_categorical_dtype(native_dtype) else promoted_dtype]

    return get_minimum_dtypes(obj)


def get_feature_names_and_dtypes(
    obj: DataLike | None,
) -> dict[Hashable, DataDTypeLike]:
    """
    Get the names and dtypes of the features in `obj` and return as dict.

    Parameters
    ----------
    obj : array-like, DataFrame, or Series, optional
        The object to get the feature names and dtypes from.

    Returns:
    --------
    dict[Hashable, DTypeLike] : dict
        The feature names and dtypes of the features in `obj`.
    """
    return {
        name: dtype
        for name, dtype in zip(
            get_feature_names(obj), get_feature_dtypes(obj), strict=True
        )
    }
