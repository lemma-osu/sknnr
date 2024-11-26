from __future__ import annotations

import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from sklearn import config_context
from sklearn.model_selection import GridSearchCV
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

TEST_YFIT_ESTIMATORS = [MSNRegressor, GNNRegressor]


def get_estimator_xfail_checks(estimator) -> dict[str, str]:
    """
    Return tests that are expected to fail with explanations.

    These are mostly due to sklearn using test data that our estimators aren't
    compatible with, e.g. 1D labels.

    Requires sklearn >= 1.6.
    """
    xfail_checks = {}

    if isinstance(estimator, GNNRegressor):
        # These checks fail due to input data constraints for the CCA ordination that
        # aren't followed by the sklearn checks.
        one_d_checks = [
            "check_estimators_dtypes",
            "check_dtype_object",
            "check_estimators_fit_returns_self",
            "check_pipeline_consistency",
            "check_estimators_overwrite_params",
            "check_fit_score_takes_y",
            "check_estimators_pickle",
            "check_regressors_train",
            "check_regressor_data_not_an_array",
            "check_regressors_no_decision_function",
            "check_supervised_y_2d",
            "check_regressors_int",
            "check_methods_sample_order_invariance",
            "check_methods_subset_invariance",
            "check_dict_unchanged",
            "check_dont_overwrite_parameters",
            "check_fit_idempotent",
            "check_fit_check_is_fitted",
            "check_n_features_in",
            "check_fit2d_predict1d",
            "check_fit2d_1sample",
            "check_estimators_nan_inf",
        ]

        row_sum_checks = [
            "check_regressor_multioutput",
            "check_readonly_memmap_input",
            "check_n_features_in_after_fitting",
            "check_f_contiguous_array_estimator",
        ]

        xfail_checks.update(
            {
                **{check: "CCA requires 2D y arrays." for check in one_d_checks},
                **{
                    check: "Row sums must be greater than 0."
                    for check in row_sum_checks
                },
            }
        )

    if isinstance(estimator, (GNNRegressor, MSNRegressor)):
        # These checks fail because the transformed estimators store the number of
        # transformed features rather than raw input features as expected by sklearn.
        n_features_in_checks = [
            "check_n_features_in_after_fitting",
            "check_n_features_in",
        ]

        xfail_checks.update(
            {
                check: "Estimator stores transformed n_features_in_"
                for check in n_features_in_checks
            }
        )

    return xfail_checks


@pytest.fixture()
def X_y_yfit() -> tuple[NDArray, NDArray, NDArray]:
    """Return X, y, and y_fit arrays for testing y_fit compatible estimators."""
    X, y = load_moscow_stjoes(return_X_y=True)
    # Arbitrary split with a constant to prevent zero sum rows
    y_fit = y[:, 10:] + 0.1
    y = y[:, :10] + 0.1
    return X, y, y_fit


@pytest.mark.filterwarnings("ignore:divide by zero encountered")
@parametrize_with_checks(
    [cls() for cls in TEST_ESTIMATORS],
    expected_failed_checks=get_estimator_xfail_checks,
)
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

    # Make sure that `list.index()` is not accidentally stored
    estimator.fit(moscow.data.tolist(), moscow.target)
    assert not hasattr(estimator, "dataframe_index_in_")

    estimator.fit(X_df, moscow.target)
    assert_array_equal(estimator.dataframe_index_in_, moscow.index)

    # Run k=1 so that each record in X_df returns itself as the neighbor
    idx = estimator.kneighbors(X_df, return_distance=False, return_dataframe_index=True)
    assert_array_equal(idx.ravel(), moscow.index)


@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_estimators_support_lists(estimator):
    """All estimators should fit and predict data stored as lists."""
    X, y = load_moscow_stjoes(return_X_y=True)
    estimator = estimator().fit(X.tolist(), y.tolist())
    estimator.predict(X.tolist())


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
    with config_context(transform_output=output_mode):
        sknnr_type = type(estimator.fit(X, y).predict(X))
        ref_type = type(ref_estimator.fit(X, y).predict(X))

    assert sknnr_type is ref_type  # noqa: E721


@pytest.mark.parametrize("estimator", TEST_YFIT_ESTIMATORS)
def test_yfit_is_stored(estimator, X_y_yfit):
    """Test that y_fit is stored when passed."""
    X, y, y_fit = X_y_yfit

    est = estimator().fit(X, y)
    assert est.y_fit_ is None
    est.fit(X, y, y_fit=y_fit)
    assert_array_equal(est.y_fit_, y_fit)


@pytest.mark.parametrize("estimator", TEST_YFIT_ESTIMATORS)
def test_yfit_affects_prediction(estimator, X_y_yfit):
    """Test that y_fit affects predictions when passed."""
    X, y, y_fit = X_y_yfit

    est = estimator()
    with_y_fit_pred = est.fit(X, y, y_fit=y_fit).independent_prediction_
    without_y_fit_pred = est.fit(X, y).independent_prediction_

    with pytest.raises(AssertionError):
        assert_array_equal(with_y_fit_pred, without_y_fit_pred)


@pytest.mark.parametrize("estimator", TEST_ESTIMATORS)
def test_gridsearchcv(estimator, X_y_yfit):
    """Test that GridSearchCV works with all estimators."""
    X, y, _ = X_y_yfit

    param_grid = {"n_neighbors": [1, 3]}
    gs = GridSearchCV(estimator(), param_grid=param_grid, cv=2)
    gs.fit(X, y)
    gs.predict(X)
