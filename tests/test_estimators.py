import pytest
from numpy.testing import assert_array_equal
from sklearn import set_config
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError

from sknnr import (
    EuclideanKNNRegressor,
    GNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    RawKNNRegressor,
)
from sknnr.datasets import load_moscow_stjoes

TEST_ESTIMATORS = [
    RawKNNRegressor,
    EuclideanKNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    GNNRegressor,
]


@pytest.mark.xfail(reason="Incrementally fixing.")
@parametrize_with_checks([cls() for cls in TEST_ESTIMATORS])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_estimators_raise_notfitted_kneighbors(estimator):
    """Attempting to call kneighbors on an unfitted estimator should raise."""
    X, y = load_moscow_stjoes(return_X_y=True)
    with pytest.raises(NotFittedError):
        estimator().kneighbors(X)


@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_estimators_raise_notfitted_predict(estimator):
    """Attempting to call predict on an unfitted estimator should raise."""
    X, y = load_moscow_stjoes(return_X_y=True)
    with pytest.raises(NotFittedError):
        estimator().predict(X)


@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_estimators_support_continuous_multioutput(estimator):
    """All estimators should fit and predict continuous multioutput data."""
    X, y = load_moscow_stjoes(return_X_y=True)
    estimator = estimator()
    estimator.fit(X, y)
    estimator.predict(X)


@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_estimators_support_dataframe_indexes(estimator):
    """All estimators should store and return dataframe indexes."""
    estimator = estimator(n_neighbors=1)
    moscow = load_moscow_stjoes()
    X_df, _ = load_moscow_stjoes(as_frame=True, return_X_y=True)

    estimator.fit(moscow.data, moscow.target)
    with pytest.raises(NotFittedError, match="fitted with a dataframe"):
        estimator.kneighbors(return_dataframe_index=True)

    estimator.fit(X_df, moscow.target)
    assert_array_equal(estimator.dataframe_index_in_, moscow.index)

    # Run k=1 so that each record in X_df returns itself as the neighbor
    idx = estimator.kneighbors(X_df, return_distance=False, return_dataframe_index=True)
    assert_array_equal(idx.ravel(), moscow.index)


@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_estimators_support_dataframes(estimator):
    """All estimators should fit and predict data stored as dataframes."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    estimator = estimator().fit(X, y)
    estimator.predict(X)

    assert_array_equal(getattr(estimator, "feature_names_in_", None), X.columns)


@pytest.mark.parametrize("fit_names", [True, False])
@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_estimators_warn_for_missing_features(estimator, fit_names):
    """All estimators should warn when fitting and predicting feature names mismatch."""
    estimator = estimator()
    X, y = load_moscow_stjoes(return_X_y=True)
    X_df, _ = load_moscow_stjoes(return_X_y=True, as_frame=True)

    if fit_names:
        msg = "fitted with feature names"
        fit_X, predict_X = X_df, X
    else:
        msg = "fitted without feature names"
        fit_X, predict_X = X, X_df

    with pytest.warns(UserWarning, match=msg):
        estimator.fit(fit_X, y)
        estimator.predict(predict_X)


@pytest.mark.parametrize("output_mode", ["default", "pandas"])
@pytest.mark.parametrize("x_type", ["array", "dataframe"])
@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_estimator_output_type_consistency(output_mode, x_type, estimator):
    """Test that output types are consistent with an sklearn estimator."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=x_type == "dataframe")
    estimator = estimator()
    ref_estimator = KNeighborsRegressor()

    # Transformer config should not affect estimator output
    set_config(transform_output=output_mode)

    sknnr_type = type(estimator.fit(X, y).predict(X))
    ref_type = type(ref_estimator.fit(X, y).predict(X))

    assert sknnr_type is ref_type  # noqa: E721
