import numpy as np
import pandas as pd
import pytest

from sknnr.utils import (
    get_feature_dtypes,
    get_feature_names,
    is_dataframe_like,
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
    assert get_feature_dtypes(obj)[0] == expected


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (np.array([True, 3], dtype=object), ([np.int64], True)),
        (np.array([1, 3.0], dtype=object), ([np.float64], True)),
        (np.array([1, "3"], dtype=object), ([np.dtype("<U21")], True)),
        (np.array([1.0, "3"], dtype=object), ([np.dtype("<U32")], True)),
        (pd.Series([1, "mixed"]), ([np.dtype("<U21")], True)),
    ],
)
def test_promoted_feature_dtypes(obj, expected):
    """Test get_feature_dtypes returns expected results when mixed data is present."""
    assert get_feature_dtypes(obj) == expected


@pytest.mark.parametrize("mixed_value", [True, 1, "mixed"])
def test_feature_dtypes_warns_with_mixed_dtypes(mixed_value):
    """Test that get_feature_dtypes raises a warning when an array has mixed dtypes."""
    arr = np.random.default_rng().random((10, 2), dtype=float)
    arr = arr.astype(object)
    arr[-1, 0] = mixed_value
    with pytest.warns(UserWarning, match=r"Column \d+ has mixed types"):
        get_feature_dtypes(arr)
