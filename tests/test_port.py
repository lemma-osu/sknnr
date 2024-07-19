import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.metrics import r2_score

from sknnr import (
    EuclideanKNNRegressor,
    GNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    RawKNNRegressor,
)

from .datasets import load_moscow_stjoes_results

ESTIMATOR_RESULTS = {
    "raw": RawKNNRegressor,
    "euclidean": EuclideanKNNRegressor,
    "mahalanobis": MahalanobisKNNRegressor,
    "gnn": GNNRegressor,
    "msn": MSNRegressor,
}


def yaimpute_weights(d):
    return 1.0 / (1.0 + d)


def estimator_does_not_support_constrained_method(
    result, *, constrained_method, **kwargs
):
    _, estimator = result
    return (
        constrained_method is not None
        and not hasattr(estimator(), "constrained_method")
    ) or (constrained_method is None and hasattr(estimator(), "constrained_method"))


def estimator_does_not_support_n_components(result, *, n_components, **kwargs):
    _, estimator = result
    return n_components is not None and not hasattr(estimator(), "n_components")


def estimator_does_not_support_parametrization(result, **kwargs):
    return estimator_does_not_support_constrained_method(
        result, **kwargs
    ) or estimator_does_not_support_n_components(result, **kwargs)


@pytest.mark.uncollect_if(func=estimator_does_not_support_parametrization)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize(
    "constrained_method", [None, "cca", "rda"], ids=["default", "cca", "rda"]
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
def test_kneighbors(result, constrained_method, n_components):
    """Test that the ported estimators identify the correct neighbors and distances."""
    method, estimator = result
    method = method if constrained_method is None else f"{method}_{constrained_method}"
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)

    hyperparams = dict(n_neighbors=5)
    hyperparams.update(
        {"constrained_method": constrained_method}
        if hasattr(estimator(), "constrained_method")
        else {}
    )
    hyperparams.update(
        {"n_components": n_components} if hasattr(estimator(), "n_components") else {}
    )

    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    dist, nn = est.kneighbors(return_dataframe_index=True)
    assert_array_equal(nn, dataset.ref_neighbors)
    assert_array_almost_equal(dist, dataset.ref_distances, decimal=3)

    dist, nn = est.kneighbors(dataset.X_test, return_dataframe_index=True)
    assert_array_equal(nn, dataset.trg_neighbors)
    assert_array_almost_equal(dist, dataset.trg_distances, decimal=3)


@pytest.mark.uncollect_if(func=estimator_does_not_support_parametrization)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize(
    "constrained_method", [None, "cca", "rda"], ids=["default", "cca", "rda"]
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("reference", [True, False], ids=["reference", "target"])
def test_predict(result, constrained_method, n_components, weighted, reference):
    """Test that the ported estimators predict the correct values."""
    method, estimator = result
    method = method if constrained_method is None else f"{method}_{constrained_method}"
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

    hyperparams = dict(n_neighbors=5, weights=weights)
    hyperparams.update(
        {"constrained_method": constrained_method}
        if hasattr(estimator(), "constrained_method")
        else {}
    )
    hyperparams.update(
        {"n_components": n_components} if hasattr(estimator(), "n_components") else {}
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    pred = est.independent_prediction_ if reference else est.predict(dataset.X_test)
    assert_array_almost_equal(pred, expected_pred, decimal=3)


@pytest.mark.uncollect_if(func=estimator_does_not_support_parametrization)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize(
    "constrained_method", [None, "cca", "rda"], ids=["default", "cca", "rda"]
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_score_independent(result, constrained_method, n_components, weighted):
    """Test that the ported estimators produce the correct score."""
    method, estimator = result
    method = method if constrained_method is None else f"{method}_{constrained_method}"
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)

    weights = yaimpute_weights if weighted else None
    predicted = (
        dataset.ref_predicted_weighted if weighted else dataset.ref_predicted_unweighted
    )
    expected_score = r2_score(dataset.y_train, predicted)

    hyperparams = dict(n_neighbors=5, weights=weights)
    hyperparams.update(
        {"constrained_method": constrained_method}
        if hasattr(estimator(), "constrained_method")
        else {}
    )
    hyperparams.update(
        {"n_components": n_components} if hasattr(estimator(), "n_components") else {}
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)
    assert est.independent_score_ == pytest.approx(expected_score, abs=0.001)
