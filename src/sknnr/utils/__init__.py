from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from warnings import warn

import numpy as np


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
    if obj is None:
        return []
    if is_dataframe_like(obj) and len(obj) > 0:
        return list(obj.columns)
    if is_series_like(obj) and len(obj) > 0:
        return ["0"] if obj.name is None else [obj.name]
    obj = np.asarray(obj, dtype=object)
    if len(obj) == 0:
        return []
    if obj.ndim == 1:
        obj = obj.reshape(-1, 1)
    return [str(i) for i in range(obj.shape[1])]


def get_feature_dtypes(obj) -> tuple[list[Any], bool]:
    """
    Get the types of the features in `obj`.

    For numpy arrays, 2D lists, and tuples, the dtype is represented as the
    smallest type that can represent all elements in that feature. When
    mixed element data types are found in a feature, a warning is emitted.

    Parameters
    ----------
    obj : array-like, DataFrame, or Series
        The object to get the feature dtypes from.

    Returns:
    --------
    list of dtypes : list[Any]
        The dtypes of the features in `obj`.
    upcast_required : bool
        When `obj` is represented as a numpy array and there are mixed types
        in one or more features, the `upcast_required` flag is set to `True`,
        otherwise it is `False`.
    """
    if obj is None:
        return [], False
    if is_dataframe_like(obj) and len(obj) > 0:
        return getattr(obj.dtypes, "values", obj.dtypes), False
    if is_series_like(obj) and len(obj) > 0 and obj.dtype != np.dtype("O"):
        return [obj.dtype], False
    obj = np.asarray(obj, dtype=object)
    if len(obj) == 0:
        return [], False
    if obj.ndim == 1:
        obj = obj.reshape(-1, 1)

    # Emit a warning if there are mixed dtypes in any column
    upcast_required = False
    for i in range(obj.shape[1]):
        unique_types = {type(x) for x in obj[:, i]}
        if len(unique_types) > 1:
            upcast_required = True
            warn(
                f"Column {i} has mixed types: {unique_types}. "
                "This may lead to unexpected behavior.",
                category=UserWarning,
                stacklevel=3,
            )

    return [
        np.asarray(obj[:, i].tolist()).dtype for i in range(obj.shape[1])
    ], upcast_required


def get_feature_names_and_dtypes(obj: Sequence) -> tuple[dict[str, Any], bool]:
    """
    Get the names and dtypes of the features in `obj` and return as dict.

    Parameters
    ----------
    obj : array-like, DataFrame, or Series
        The object to get the feature names and dtypes from.

    Returns:
    --------
    dict[str, Any]:
        The feature names and dtypes of the features in `obj`.
    upcast_required : bool
        When `obj` is represented as a numpy array and there are mixed types
        in one or more features, the `upcast_required` flag is set to `True`,
        otherwise it is `False`.
    """
    feature_dtypes, upcast_required = get_feature_dtypes(obj)
    return {
        name: dtype for name, dtype in zip(get_feature_names(obj), feature_dtypes)
    }, upcast_required
