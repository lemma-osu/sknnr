import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

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


@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
def test_kneighbors(result):
    """Test that the ported estimators identify the correct neighbors and distances."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method)

    est = estimator(n_neighbors=5).fit(dataset.X_train, dataset.y_train)

    dist, nn = est.kneighbors(return_dataframe_index=True)
    assert_array_equal(nn, dataset.ref_neighbors)
    assert_array_almost_equal(dist, dataset.ref_distances, decimal=3)

    dist, nn = est.kneighbors(dataset.X_test, return_dataframe_index=True)
    assert_array_equal(nn, dataset.trg_neighbors)
    assert_array_almost_equal(dist, dataset.trg_distances, decimal=3)


@pytest.mark.parametrize(
    "result", ESTIMATOR_RESULTS.items(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_predict(result, weighted):
    """Test that the ported estimators predict the correct values."""
    method, estimator = result
    dataset = load_moscow_stjoes_results(method=method)

    weights = yaimpute_weights if weighted else None
    trg_predicted = (
        dataset.trg_predicted_weighted if weighted else dataset.trg_predicted_unweighted
    )

    est = estimator(n_neighbors=5, weights=weights).fit(
        dataset.X_train, dataset.y_train
    )
    prd = est.predict(dataset.X_test)
    assert_array_almost_equal(prd, trg_predicted, decimal=3)
