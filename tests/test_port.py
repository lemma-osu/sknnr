import inspect

import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.metrics import r2_score

from sknnr import (
    EuclideanKNNRegressor,
    GNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    RawKNNRegressor,
    RFNNRegressor,
)

from .datasets import load_moscow_stjoes_results

ESTIMATOR_RESULTS = {
    "raw": RawKNNRegressor,
    "euclidean": EuclideanKNNRegressor,
    "mahalanobis": MahalanobisKNNRegressor,
    "gnn": GNNRegressor,
    "msn": MSNRegressor,
    "randomForest": RFNNRegressor,
}


def yaimpute_weights(d):
    return 1.0 / (1.0 + d)


def estimator_does_not_support_n_components(result, n_components, **kwargs):
    _, estimator = result
    return n_components is not None and not hasattr(estimator(), "n_components")


def get_default_hyperparams(estimator, **kwargs) -> dict:
    """Return valid parameters for the given estimator, including common defaults."""
    default_params = dict(
        n_neighbors=5,
        random_state=42,
        **kwargs,
    )

    valid_params = inspect.signature(estimator).parameters
    return {k: v for k, v in default_params.items() if k in valid_params}


@pytest.mark.uncollect_if(func=estimator_does_not_support_n_components)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
def test_kneighbors(result, n_components):
    """Test that the ported estimators identify the correct neighbors and distances."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)

    hyperparams = get_default_hyperparams(estimator, n_components=n_components)
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    dist, nn = est.kneighbors(return_dataframe_index=True)
    assert_array_equal(nn, dataset.ref_neighbors)
    assert_array_almost_equal(dist, dataset.ref_distances, decimal=3)

    dist, nn = est.kneighbors(dataset.X_test, return_dataframe_index=True)
    assert_array_equal(nn, dataset.trg_neighbors)
    assert_array_almost_equal(dist, dataset.trg_distances, decimal=3)


@pytest.mark.uncollect_if(func=estimator_does_not_support_n_components)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("reference", [True, False], ids=["reference", "target"])
def test_predict(result, n_components, weighted, reference):
    """Test that the ported estimators predict the correct values."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)

    weights = yaimpute_weights if weighted else None
    if weighted and reference:
        expected_pred = dataset.ref_predicted_weighted
    elif weighted and not reference:
        expected_pred = dataset.trg_predicted_weighted
    elif not weighted and reference:
        expected_pred = dataset.ref_predicted_unweighted
    else:
        expected_pred = dataset.trg_predicted_unweighted

    hyperparams = get_default_hyperparams(
        estimator, n_components=n_components, weights=weights
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    pred = est.independent_prediction_ if reference else est.predict(dataset.X_test)
    assert_array_almost_equal(pred, expected_pred, decimal=3)


@pytest.mark.uncollect_if(func=estimator_does_not_support_n_components)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("reference", [True, False], ids=["reference", "target"])
def test_kneighbors_regressions(ndarrays_regression, result, n_components, reference):
    """Test that the ported estimators identify the correct neighbors and distances."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)

    hyperparams = get_default_hyperparams(estimator, n_components=n_components)
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    if reference:
        dist, nn = est.kneighbors(return_dataframe_index=True)
    else:
        dist, nn = est.kneighbors(dataset.X_test, return_dataframe_index=True)
    ndarrays_regression.check(dict(dist=dist, nn=nn))


@pytest.mark.uncollect_if(func=estimator_does_not_support_n_components)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("reference", [True, False], ids=["reference", "target"])
def test_predict_regressions(
    ndarrays_regression, result, n_components, weighted, reference
):
    """Test that the ported estimators predict the correct values."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)

    weights = yaimpute_weights if weighted else None

    hyperparams = get_default_hyperparams(
        estimator, n_components=n_components, weights=weights
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    if reference:
        pred = est.independent_prediction_
        score = est.independent_score_
        ndarrays_regression.check(dict(pred=pred, score=score))
    else:
        pred = est.predict(dataset.X_test)
        ndarrays_regression.check(dict(pred=pred))


@pytest.mark.uncollect_if(func=estimator_does_not_support_n_components)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_score_independent(result, n_components, weighted):
    """Test that the ported estimators produce the correct score."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)
    weights = yaimpute_weights if weighted else None
    predicted = (
        dataset.ref_predicted_weighted if weighted else dataset.ref_predicted_unweighted
    )
    expected_score = r2_score(dataset.y_train, predicted)

    hyperparams = get_default_hyperparams(
        estimator, n_components=n_components, weights=weights
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)
    assert est.independent_score_ == pytest.approx(expected_score, abs=0.001)
