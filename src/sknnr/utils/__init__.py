from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def is_dataframe_like(obj: Any) -> bool:
    """
    Check if `obj` is a dataframe-like structure.  This will evaluate to `True`
    for pandas and polars DataFrames without the need to import those packages
    explicitly.
    """
    return hasattr(obj, "columns") and hasattr(obj, "dtypes")


def is_series_like(obj: Any) -> bool:
    """
    Check if `obj` is a series-like structure.  This will evaluate to `True`
    for pandas and polars Series without the need to import those packages
    explicitly.
    """
    return hasattr(obj, "name") and hasattr(obj, "dtype")


def is_number_like_type(t: Any) -> bool:
    """
    Check if `t` is a number-like type.  For most types, np.issubdtype will
    correctly identify the type.  For pandas extension types, we can check the
    kind of the type.
    """
    try:
        return np.issubdtype(t, np.number)
    except TypeError:
        try:
            return t.kind in "iuf"
        except AttributeError as err:
            msg = f"Unsupported type {t}"
            raise TypeError(msg) from err


def is_nan_like(x: Any) -> bool:
    """Check if `x` is NaN-like."""
    return bool(
        x is None
        or (isinstance(x, float) and np.isnan(x))
        or x.__class__.__name__ == "NAType"
    )


def get_feature_names(obj) -> list[str]:
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
    obj = np.asarray(obj, dtype=object)
    if obj.ndim == 1:
        obj = obj.reshape(-1, 1)
    return [str(i) for i in range(obj.shape[1])]


def get_minimum_dtypes(obj: Any) -> list[np.dtype]:
    """
    Return the smallest numpy dtype that can accommodate all data for each
    column in obj.
    """
    obj = np.asarray(obj, dtype=object)
    if obj.ndim == 1:
        obj = obj.reshape(-1, 1)
    return [np.asarray(obj[:, i].tolist()).dtype for i in range(obj.shape[1])]


def get_feature_dtypes(obj) -> list[np.dtype | pd.CategoricalDtype]:
    """
    Get numpy dtypes of the features in `obj`, promoting dtypes as necessary to
    the smallest type that can accommodate all data elements in that feature.
    The exception to this is when a pd.CategoricalDtype feature is present,
    retain this dtype.

    Parameters
    ----------
    obj : array-like, DataFrame, or Series
        The object from which to get the feature dtypes.

    Returns:
    --------
    list of dtypes : list[np.dtype | pd.CategoricalDtype]
        The dtypes (either numpy or pd.CategoricalDtype) of the features in `obj`.
    """
    if obj is None or len(np.asarray(obj)) == 0:
        return []

    if is_dataframe_like(obj):
        native_dtypes = getattr(obj.dtypes, "values", obj.dtypes)
        promoted_dtypes = get_minimum_dtypes(obj)
        return [
            p_dtype if str(n_dtype) != "category" else n_dtype
            for p_dtype, n_dtype in zip(promoted_dtypes, native_dtypes)
        ]
    if is_series_like(obj):
        native_dtype = obj.dtype
        promoted_dtype = get_minimum_dtypes(obj)[0]
        return [promoted_dtype if str(native_dtype) != "category" else native_dtype]

    return get_minimum_dtypes(obj)


def get_feature_names_and_dtypes(
    obj: Sequence,
) -> dict[str, np.dtype | pd.CategoricalDtype]:
    """
    Get the names and dtypes of the features in `obj` and return as dict.

    Parameters
    ----------
    obj : array-like, DataFrame, or Series
        The object to get the feature names and dtypes from.

    Returns:
    --------
    dict[str, np.dtype | pd.CategoricalDtype] : dict
        The feature names and dtypes of the features in `obj`.
    """
    return {
        name: dtype
        for name, dtype in zip(get_feature_names(obj), get_feature_dtypes(obj))
    }
