from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def is_dataframe_like(obj: Any) -> bool:
    """
    Check if `obj` is a dataframe-like structure.  This will evaluate to `True`
    for pandas and polars DataFrames without the need to import those packages
    explictly.
    """
    return hasattr(obj, "columns") and hasattr(obj, "dtypes")


def is_series_like(obj: Any) -> bool:
    """
    Check if `obj` is a series-like structure.  This will evaluate to `True`
    for pandas and polars Series without the need to import those packages
    explictly.
    """
    return hasattr(obj, "name") and hasattr(obj, "dtype")


def get_feature_names(obj) -> list[str]:
    """
    Get the names of the features in `obj`. If no names are found, return
    a list of strings with the feature index.
    """
    if obj is None:
        return []
    if is_dataframe_like(obj):
        return list(obj.columns)
    if is_series_like(obj):
        return ["0"] if obj.name is None else [obj.name]
    obj = np.asarray(obj, dtype=object)
    if len(obj) == 0:
        return []
    if obj.ndim == 1:
        obj = obj.reshape(-1, 1)
    return [str(i) for i in range(obj.shape[1])]


def get_feature_dtypes(obj) -> list[Any]:
    """
    Get the types of the features in `obj`. For numpy arrays, 2D lists, and tuples,
    dtype is inferred from the first row's element's dtype.
    """
    if obj is None:
        return []
    if is_dataframe_like(obj):
        return obj.dtypes.values if hasattr(obj.dtypes, "values") else obj.dtypes
    if is_series_like(obj):
        return [obj.dtype]
    obj = np.asarray(obj, dtype=object)
    if len(obj) == 0:
        return []
    if obj.ndim == 1:
        obj = obj.reshape(-1, 1)
    return [type(obj[0, k]) for k in range(obj.shape[1])]


def get_feature_names_and_dtypes(obj: Sequence) -> dict[str, Any]:
    """Get the names and dtypes of the features in `obj` and return as dict."""
    return {
        name: dtype
        for name, dtype in zip(get_feature_names(obj), get_feature_dtypes(obj))
    }
