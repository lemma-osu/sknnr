from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import TypeGuard

    from ..types import DataFrameLike, DTypeLike, SeriesLike


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


def is_number_like_type(t: DTypeLike) -> bool:
    """
    Check if `t` is a number-like type.  For most types, np.issubdtype will
    correctly identify the type.  For pandas extension types, we can check the
    kind of the type.
    """
    try:
        return np.issubdtype(t, np.number)  # type: ignore[arg-type]
    except TypeError as e:
        if kind := getattr(t, "kind", None):
            return kind in "iuf"
        raise TypeError(f"Unsupported type {t}") from e


def is_nan_like(x: object) -> bool:
    """Check if `x` is NaN-like, including None, np.nan, and pd.NA values."""
    return bool(
        x is None
        or (isinstance(x, float) and np.isnan(x))
        or x.__class__.__name__ == "NAType"
    )


def get_feature_names(
    obj: None | Sequence | DataFrameLike | SeriesLike,
) -> list[Hashable]:
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


def get_minimum_dtypes(obj: Sequence | DataFrameLike | SeriesLike) -> list[DTypeLike]:
    """
    Return the smallest numpy dtype that can accommodate all data for each
    column in obj.
    """
    arr = np.asarray(obj, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return [np.asarray(arr[:, i].tolist()).dtype for i in range(arr.shape[1])]


def get_feature_dtypes(
    obj: None | Sequence | DataFrameLike | SeriesLike,
) -> list[DTypeLike]:
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
            p_dtype if str(n_dtype) != "category" else n_dtype
            for p_dtype, n_dtype in zip(promoted_dtypes, native_dtypes, strict=True)
        ]
    if is_series_like(obj):
        native_dtype = obj.dtype
        promoted_dtype = get_minimum_dtypes(obj)[0]
        return [promoted_dtype if str(native_dtype) != "category" else native_dtype]

    return get_minimum_dtypes(obj)


def get_feature_names_and_dtypes(
    obj: None | Sequence | DataFrameLike | SeriesLike,
) -> dict[Hashable, DTypeLike]:
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
