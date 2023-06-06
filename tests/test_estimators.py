from typing import List

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn import set_config
from sklearn.neighbors import KNeighborsRegressor

# from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError

from sknnr import (
    EuclideanKNNRegressor,
    GNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    RawKNNRegressor,
)
from sknnr._base import IDNeighborsRegressor


def get_kneighbor_estimator_classes() -> List[IDNeighborsRegressor]:
    """
    Return classes of all supported IDNeighborsRegressor estimators.
    """
    return [
        RawKNNRegressor,
        EuclideanKNNRegressor,
        MahalanobisKNNRegressor,
        MSNRegressor,
        GNNRegressor,
    ]


def get_kneighbor_estimator_instances() -> List[IDNeighborsRegressor]:
    """
    Return instances of all supported IDNeighborsRegressor estimators.
    """
    return [cls() for cls in get_kneighbor_estimator_classes()]


# Note: This will run all the sklearn estimator checks. It's going to take quite a bit
# of work to get these all passing, and it's possible we just won't be able to do it
# while maintaining all the features we need.
# @parametrize_with_checks(get_kneighbor_estimator_instances())
# def test_sklearn_compatibile_estimators(estimator, check):
#     check(estimator)


@pytest.mark.parametrize("estimator", get_kneighbor_estimator_classes())
def test_estimators_raise_notfitted_kneighbors(estimator, moscow_euclidean):
    """Attempting to call kneighbors on an unfitted estimator should raise."""
    with pytest.raises(NotFittedError):
        estimator().kneighbors(moscow_euclidean.X)


@pytest.mark.parametrize("estimator", get_kneighbor_estimator_classes())
def test_estimators_raise_notfitted_predict(estimator, moscow_euclidean):
    """Attempting to call predict on an unfitted estimator should raise."""
    with pytest.raises(NotFittedError):
        estimator().predict(moscow_euclidean.X)


@pytest.mark.parametrize("estimator", get_kneighbor_estimator_classes())
def test_estimators_support_continuous_multioutput(estimator, moscow_euclidean):
    """All estimators should fit and predict continuous multioutput data."""
    estimator = estimator()
    estimator.fit(moscow_euclidean.X, moscow_euclidean.y)
    estimator.predict(moscow_euclidean.X)


@pytest.mark.parametrize("estimator", get_kneighbor_estimator_classes())
def test_estimators_support_dataframes(estimator, moscow_euclidean):
    """All estimators should fit and predict data stored as dataframes."""
    estimator = estimator()
    num_features = moscow_euclidean.X.shape[1]
    feature_names = [f"col_{i}" for i in range(num_features)]

    X_df = pd.DataFrame(moscow_euclidean.X, columns=feature_names)
    y_df = pd.DataFrame(moscow_euclidean.y)

    estimator.fit(X_df, y_df)
    estimator.predict(X_df)
    assert_array_equal(estimator.feature_names_in_, feature_names)


@pytest.mark.parametrize("estimator", get_kneighbor_estimator_classes())
def test_estimators_warn_for_missing_features(estimator, moscow_euclidean):
    """All estimators should warn when fitting and predicting feature names mismatch."""
    estimator = estimator()
    num_features = moscow_euclidean.X.shape[1]
    feature_names = [f"col_{i}" for i in range(num_features)]

    X = moscow_euclidean.X
    y = moscow_euclidean.y
    X_df = pd.DataFrame(X, columns=feature_names)

    # Fit without feature names, predict with feature names
    with pytest.warns(UserWarning, match="fitted without feature names"):
        estimator.fit(X, y)
        estimator.predict(X_df)

    # Fit with feature names, predict without feature names
    with pytest.warns(UserWarning, match="fitted with feature names"):
        estimator.fit(X_df, y)
        estimator.predict(X)


@pytest.mark.parametrize("output_mode", ["default", "pandas"])
@pytest.mark.parametrize("x_type", ["array", "dataframe"])
@pytest.mark.parametrize("estimator", get_kneighbor_estimator_classes())
def test_estimator_output_type_consistency(
    output_mode, x_type, estimator, moscow_euclidean
):
    """Test that output types are consistent with an sklearn estimator."""
    X, y = moscow_euclidean.X, moscow_euclidean.y
    X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    x = X if x_type == "array" else X_df

    estimator = estimator()
    ref_estimator = KNeighborsRegressor()

    # Transformer config should not affect estimator output
    set_config(transform_output=output_mode)

    sknnr_type = type(estimator.fit(x, y).predict(x))
    ref_type = type(ref_estimator.fit(x, y).predict(x))

    assert sknnr_type is ref_type  # noqa: E721
