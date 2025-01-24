import pytest
from numpy import corrcoef
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.stats import spearmanr
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

    hyperparams = dict(n_neighbors=5, weights=weights)
    hyperparams.update(
        {"n_components": n_components} if hasattr(estimator(), "n_components") else {}
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    pred = est.independent_prediction_ if reference else est.predict(dataset.X_test)
    assert_array_almost_equal(pred, expected_pred, decimal=3)


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

    hyperparams = dict(n_neighbors=5, weights=weights)
    hyperparams.update(
        {"n_components": n_components} if hasattr(estimator(), "n_components") else {}
    )
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)
    assert est.independent_score_ == pytest.approx(expected_score, abs=0.001)


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("reference", [True, False], ids=["reference", "target"])
def test_high_correlation_rfnn(weighted, reference):
    """
    Test that predictions for RFNN estimator using yaImpute and sklearn
    highly correlate.  Currently failing.
    """
    dataset = load_moscow_stjoes_results(method="randomForest", n_components=None)
    weights = yaimpute_weights if weighted else None

    hyperparams = dict(n_neighbors=5, weights=weights)
    rpy2_est = RFNNRegressor(method="rpy2", **hyperparams).fit(
        dataset.X_train, dataset.y_train
    )
    sklearn_est = RFNNRegressor(method="sklearn", **hyperparams).fit(
        dataset.X_train, dataset.y_train
    )

    rpy2_pred = (
        rpy2_est.independent_prediction_
        if reference
        else rpy2_est.predict(dataset.X_test)
    )
    sklearn_pred = (
        sklearn_est.independent_prediction_
        if reference
        else sklearn_est.predict(dataset.X_test)
    )
    corr = [
        float(corrcoef(rpy2_pred[:, i], sklearn_pred[:, i])[0, 1])
        for i in range(rpy2_pred.shape[1])
        if rpy2_pred[:, i].std() > 0 and sklearn_pred[:, i].std() > 0
    ]
    print(corr)
    assert all(c > 0.99 for c in corr)


def test_kneighbors_rfnn():
    """
    Test that plots from different implementations of RF-NN share *most*
    of the same neighbors. Currently failing.
    """
    dataset = load_moscow_stjoes_results(method="randomForest", n_components=None)

    hyperparams = dict(n_neighbors=13)
    rpy2_est = RFNNRegressor(method="rpy2", **hyperparams).fit(
        dataset.X_train, dataset.y_train
    )
    sklearn_est = RFNNRegressor(method="sklearn", **hyperparams).fit(
        dataset.X_train, dataset.y_train
    )

    _, rpy2_nn = rpy2_est.kneighbors(return_dataframe_index=True)
    _, sklearn_nn = sklearn_est.kneighbors(return_dataframe_index=True)

    def footrule_distance(list1, list2):
        """
        The Footrule Distance measures how far elements in one list are from
        their corresponding positions in another list. It calculates the sum of absolute
        positional differences.
        """
        index_map = {int(val): i for i, val in enumerate(list2)}
        not_in_list = len(list2) + 1
        diffs = []
        for i, val in enumerate(list1):
            if int(val) not in index_map:
                diffs.append(abs(i - not_in_list))
            else:
                diffs.append(abs(i - index_map[int(val)]))
        return sum(diffs)

    footrule_distances = []
    for i in range(rpy2_nn.shape[0]):
        fd = footrule_distance(rpy2_nn[i], sklearn_nn[i])
        print(rpy2_nn[i])
        print(sklearn_nn[i])
        print(spearmanr(rpy2_nn[i], sklearn_nn[i]))
        print(fd)
        footrule_distances.append(fd)

    assert all(fd < 10 for fd in footrule_distances)
