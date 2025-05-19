import numpy as np
import pandas as pd
import pytest

from sknnr.utils import (
    get_feature_dtypes,
    get_feature_names,
    is_dataframe_like,
    is_nan_like,
    is_number_like_type,
    is_series_like,
)


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (pd.DataFrame(), True),
        (pd.DataFrame({"a": [1, 2]}), True),
        (pd.Series().to_frame(), True),
        (np.array([]), False),
        (list([]), False),
        (tuple([]), False),
        (pd.Series(), False),
        (None, False),
    ],
)
def test_is_dataframe_like(obj, expected):
    """Test is_dataframe_like returns expected results."""
    assert is_dataframe_like(obj) is expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (pd.Series(), True),
        (pd.Series({"a": [1, 2]}), True),
        (pd.DataFrame({"a": [1, 2]}).a, True),
        (np.array([]), False),
        (list([]), False),
        (tuple([]), False),
        (pd.DataFrame(), False),
        (None, False),
    ],
)
def test_is_series_like(obj, expected):
    """Test is_series_like returns expected results."""
    assert is_series_like(obj) is expected


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (int, True),
        (float, True),
        (np.int32, True),
        (np.float64, True),
        (pd.Int32Dtype(), True),
        (pd.Float64Dtype(), True),
        (np.object_, False),
        (pd.StringDtype(), False),
        (pd.CategoricalDtype(), False),
        (type(None), False),
    ],
)
def test_is_number_like_type(t, expected):
    """Test is_number_like_type returns expected results."""
    assert is_number_like_type(t) is expected


@pytest.mark.parametrize("x", [None, np.nan, float("nan"), pd.NA])
def test_is_nan_like(x):
    """Test is_nan_like returns expected results."""
    assert is_nan_like(x) is True


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (np.array([]), []),
        (np.array([[1, 2], [3, 4]]), ["0", "1"]),
        (pd.DataFrame(), []),
        (pd.DataFrame({"a": [1, 2]}), ["a"]),
        (pd.Series(), []),
        (pd.Series({"a": 1, "b": 2}), ["0"]),
        (pd.Series({"a": 1, "b": 2}, name="s"), ["s"]),
        (None, []),
    ],
)
def test_get_feature_names(obj, expected):
    """Test get_feature_names returns expected results."""
    assert get_feature_names(obj) == expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (np.array([]), []),
        (np.array([[1, 2], [3, 4]]), [np.int64, np.int64]),
        (pd.DataFrame(), []),
        (pd.DataFrame({"a": [1, 2]}), [np.int64]),
        (pd.Series(), []),
        (pd.Series([1, 2]), [np.int64]),
        (pd.Series({"a": 1, "b": 2}), [np.int64]),
        (None, []),
    ],
)
def test_get_feature_dtypes(obj, expected):
    """Test get_feature_dtypes returns expected results."""
    assert get_feature_dtypes(obj) == expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (
            pd.Series([1, 2, 3, 4], dtype=pd.CategoricalDtype()),
            [pd.CategoricalDtype(categories=[1, 2, 3, 4], ordered=False)],
        ),
        (pd.Series([1, 2, 3, 4], dtype=pd.Int64Dtype()), [np.int64]),
        (pd.Series([1, 2, 3, 4], dtype=pd.Float64Dtype()), [np.float64]),
        (pd.Series([1, 2, 3, 4], dtype=pd.StringDtype()), [np.dtype("<U1")]),
        (pd.Series([1, 1, 0, 0], dtype=pd.BooleanDtype()), [np.bool_]),
    ],
)
def test_get_feature_dtypes_handles_pandas_extension_dtypes(obj, expected):
    """
    Test get_feature_dtypes correctly converts pandas ExtensionDtypes into
    numpy dtypes for use in modeling.  Note that pd.CategoricalDtype is not
    converted to a numpy dtype, but is returned as is.
    """
    assert get_feature_dtypes(obj) == expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (np.array([True, 3], dtype=object), [np.int64]),
        (np.array([1, 3.0], dtype=object), [np.float64]),
        (np.array([1, "3"], dtype=object), [np.dtype("<U21")]),
        (np.array([1.0, "3"], dtype=object), [np.dtype("<U32")]),
        (pd.Series([1, "mixed"]), [np.dtype("<U21")]),
    ],
)
def test_promoted_feature_dtypes(obj, expected):
    """
    Test get_feature_dtypes returns correct promoted dtypes when given
    elements with mixed dtypes.
    """
    assert get_feature_dtypes(obj) == expected
