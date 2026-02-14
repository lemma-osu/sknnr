from __future__ import annotations

import numpy as np
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
    GBNNRegressor,
    GNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    RawKNNRegressor,
    RFNNRegressor,
)
from sknnr.datasets import load_moscow_stjoes

TEST_ESTIMATORS = [
    RawKNNRegressor,
    EuclideanKNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    GNNRegressor,
    RFNNRegressor,
    GBNNRegressor,
]

TEST_TRANSFORMED_ESTIMATORS = [
    EuclideanKNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    GNNRegressor,
    RFNNRegressor,
    GBNNRegressor,
]

TEST_YFIT_ESTIMATORS = [MSNRegressor, GNNRegressor, RFNNRegressor, GBNNRegressor]

TEST_TREE_BASED_ESTIMATORS = [RFNNRegressor, GBNNRegressor]


def standardize_user_forest_weights(
    forest_weights: str | list[int | float], n_forests: int
) -> NDArray[np.float64]:
    """Standardize user-supplied forest weights for comparison."""
    if isinstance(forest_weights, str) and forest_weights == "uniform":
        forest_weights_arr = np.ones(n_forests, dtype="float64") / n_forests
    else:
        forest_weights_arr = np.asarray(forest_weights, dtype="float64")
        forest_weights_arr /= np.sum(forest_weights_arr)
    return forest_weights_arr


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
            "check_fit2d_predict1d",
            "check_fit2d_1sample",
            "check_estimators_nan_inf",
            "check_positive_only_tag_during_fit",
        ]

        row_sum_checks = [
            "check_regressor_multioutput",
            "check_readonly_memmap_input",
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

    if isinstance(estimator, tuple(TEST_YFIT_ESTIMATORS)):
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


@pytest.fixture
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


@pytest.mark.parametrize("estimator", TEST_TRANSFORMED_ESTIMATORS)
def test_n_features_in(estimator, X_y_yfit):
    """
    Test that estimators store the number of transformed features.
    """
    X, y, _ = X_y_yfit

    est = estimator().fit(X, y)
    transformed_features = est.transformer_.get_feature_names_out()

    assert est.transformer_.n_features_in_ == X.shape[1]
    assert est.n_features_in_ == len(transformed_features)


@pytest.mark.parametrize(
    ("use_deterministic_ordering", "expected_idx_order"),
    [(False, [1, 0]), (True, [0, 1])],
)
def test_kneighbors_deterministic_ordering(
    use_deterministic_ordering, expected_idx_order
):
    """
    Test that the use_deterministic_ordering parameter affects the order
    of neighbors when distances are nearly identical.
    """
    X = np.array([1e-11, 1e-12, 1.0]).reshape(-1, 1)
    y = np.array([0, 1, 2])

    X_query = np.array([[0.0]])

    _, idx = (
        RawKNNRegressor(n_neighbors=2)
        .fit(X, y)
        .kneighbors(X_query, use_deterministic_ordering=use_deterministic_ordering)
    )
    assert_array_equal(idx[0], expected_idx_order)


def test_kneighbors_uses_index_difference():
    """
    Test that when distances are considered to be identical, the absolute index
    difference is used before indexes to order neighbors.
    """
    X = np.array([1e-11, 1e-12, 1.0]).reshape(-1, 1)
    y = np.array([0, 1, 2])

    # Use two identical query points which should have different
    # neighbors due to their row indexes
    X_query = np.array([[0.0], [0.0]])

    _, idx = (
        RawKNNRegressor(n_neighbors=2)
        .fit(X, y)
        .kneighbors(X_query, use_deterministic_ordering=True)
    )
    assert_array_equal(idx[0], [0, 1])
    assert_array_equal(idx[1], [1, 0])


@pytest.mark.parametrize(
    ("precision_decimals", "expected_idx_order"),
    [(8, [2, 1, 0]), (5, [1, 2, 0]), (2, [0, 1, 2])],
)
def test_kneighbors_precision_decimals(
    monkeypatch, precision_decimals, expected_idx_order
):
    """
    Test that changing DISTANCE_PRECISION_DECIMALS affects the order
    of neighbors on small precision differences.
    """
    monkeypatch.setattr(
        RawKNNRegressor, "DISTANCE_PRECISION_DECIMALS", precision_decimals
    )

    # Create features that differ by small amounts such that
    # precision_decimals falls between them
    X = np.array([1e-3, 1e-6, 1e-9, 1.0]).reshape(-1, 1)
    y = np.array([0, 1, 2, 3])

    X_query = np.array([[0.0]])

    _, idx = (
        RawKNNRegressor(n_neighbors=3)
        .fit(X, y)
        .kneighbors(X_query, use_deterministic_ordering=True)
    )
    assert_array_equal(idx[0], expected_idx_order)


@pytest.mark.parametrize("estimator", TEST_TREE_BASED_ESTIMATORS)
@pytest.mark.parametrize("forest_weights", ["uniform", [0.5, 1.5], (1.0, 2.0)])
def test_tree_estimator_handles_forest_weights(estimator, forest_weights):
    """Test tree-based estimators handle forest weights correctly."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    y = y.iloc[:, :2]

    est = estimator(forest_weights=forest_weights).fit(X, y)

    assert hasattr(est.transformer_, "tree_weights_")

    # `est.hamming_weights_` is a 1D array with length equal to the
    # total number of trees across all forests.  The sum of the weights across
    # trees in each forest should be proportional to the corresponding forest
    # weight
    calculated_forest_weights = est.hamming_weights_.reshape(
        est.transformer_.n_forests_, -1
    ).sum(axis=1)

    # Standardize the user-supplied forest weights for comparison
    forest_weights_arr = standardize_user_forest_weights(
        forest_weights, est.transformer_.n_forests_
    )

    assert np.allclose(calculated_forest_weights, forest_weights_arr, atol=1e-3)


@pytest.mark.parametrize("forest_weights", ["uniform", [0.5, 1.5], (1.0, 2.0)])
def test_gbnn_multiclass_weights(forest_weights):
    """Test GBNN in multi-class mode handles forest weights correctly."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    y_fit = y[["Total_BA"]].assign(
        ABGR_CLASS=np.digitize(
            y.iloc[:, 0], np.percentile(y.iloc[:, 0], [33, 66])
        ).astype(str)
    )

    est = GBNNRegressor(forest_weights=forest_weights).fit(X, y, y_fit=y_fit)
    assert hasattr(est.transformer_, "tree_weights_")

    # With GBNN multi-class classification forests, `est.hamming_weights_` is a
    # bit more complex.  It is still a 1D array, but the length is equal to
    # the sum of (n_classes * n_estimators) across all forests.  In this test,
    # with n_estimators=100 and 3 classes, the classification forest will
    # have 300 weights, and the regression forest will have 100 weights.
    # The sum of the weights across all trees in the forest should still be
    # proportional to the corresponding forest weight, but will be divided by
    # and repeated for the number of classes.

    calculated_forest_weights = est.hamming_weights_.reshape(
        -1, est.transformer_.n_estimators
    ).sum(axis=1)

    # Standardize the user-supplied forest weights for comparison
    forest_weights_arr = standardize_user_forest_weights(
        forest_weights, est.transformer_.n_forests_
    )

    # Spread and repeat the forest weights across the number of classes
    n_classes = np.asarray(est.transformer_.n_trees_per_iteration_)
    expected_weights = np.repeat(forest_weights_arr / n_classes, n_classes)
    assert np.allclose(calculated_forest_weights, expected_weights, atol=1e-3)


@pytest.mark.parametrize("estimator", TEST_TREE_BASED_ESTIMATORS)
@pytest.mark.parametrize("forest_weights", ["uniform", [0.5, 1.5], (1.0, 2.0)])
@pytest.mark.parametrize("tree_weighting_method", ["uniform", "train_improvement"])
def test_hamming_weights_sum_to_one(estimator, forest_weights, tree_weighting_method):
    """Test tree-based estimators create hamming weights that sum to 1."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    num_weights = 1 if forest_weights == "uniform" else len(forest_weights)
    y = y.iloc[:, :num_weights]

    # Set tree_weighting_method for GBNNRegressor
    kwargs = (
        {"tree_weighting_method": tree_weighting_method}
        if estimator is GBNNRegressor
        else {}
    )
    est = estimator(forest_weights=forest_weights, **kwargs).fit(X, y)
    assert np.isclose(est.hamming_weights_.sum(), 1.0)


@pytest.mark.parametrize("estimator", TEST_TREE_BASED_ESTIMATORS)
def test_tree_estimator_raises_on_invalid_forest_weights(estimator):
    """Test that tree-based estimators raise on invalid forest weights."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    y = y.iloc[:, :2]

    with pytest.raises(ValueError, match=r"Expected `forest_weights` to have length 2"):
        estimator(forest_weights=[0.5]).fit(X, y)
