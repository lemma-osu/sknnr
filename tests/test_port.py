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


def estimator_does_not_support_n_components(result, n_components, **kwargs):
    _, estimator = result
    return n_components is not None and not hasattr(estimator(), "n_components")


@pytest.mark.uncollect_if(func=estimator_does_not_support_n_components)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
def test_kneighbors(result, n_components):
    """Test that the ported estimators identify the correct neighbors and distances."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)

    hyperparams = dict(n_neighbors=5)
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


@pytest.mark.uncollect_if(func=estimator_does_not_support_n_components)
@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("training", [True, False], ids=["training", "testing"])
def test_predict(result, n_components, weighted, training):
    """Test that the ported estimators predict the correct values."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method, n_components=n_components)

    weights = yaimpute_weights if weighted else None
    validation_dataset = {
        (True, True): dataset.ref_predicted_weighted,
        (True, False): dataset.ref_predicted_unweighted,
        (False, True): dataset.trg_predicted_weighted,
        (False, False): dataset.trg_predicted_unweighted,
    }
    predicted = validation_dataset[(training, weighted)]

    hyperparams = dict(n_neighbors=5, weights=weights)
    hyperparams.update(
        {"n_components": n_components} if hasattr(estimator(), "n_components") else {}
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    prd = est.predict_independent() if training else est.predict(dataset.X_test)
    assert_array_almost_equal(prd, predicted, decimal=3)


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
    expected_score = r2_score(dataset.y_train, predicted).mean()

    hyperparams = dict(n_neighbors=5, weights=weights)
    hyperparams.update(
        {"n_components": n_components} if hasattr(estimator(), "n_components") else {}
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)
    score = est.score_independent(dataset.y_train)
    assert score == pytest.approx(expected_score, abs=0.001)
