import inspect

import pytest

from sknnr import (
    EuclideanKNNRegressor,
    GNNRegressor,
    MahalanobisKNNRegressor,
    MSNRegressor,
    RawKNNRegressor,
    RFNNRegressor,
)

from .datasets import load_moscow_stjoes_test_data

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


def estimator_does_not_support_n_components(estimator, n_components, **kwargs):
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
    "estimator", ESTIMATOR_RESULTS.values(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("reference", [True, False], ids=["reference", "target"])
def test_kneighbors(ndarrays_regression, estimator, n_components, reference):
    """Test that the ported estimators identify the correct neighbors and distances."""
    dataset = load_moscow_stjoes_test_data()

    hyperparams = get_default_hyperparams(estimator, n_components=n_components)
    est = estimator(**hyperparams).fit(dataset.X_train, dataset.y_train)

    if reference:
        dist, nn = est.kneighbors(return_dataframe_index=True)
    else:
        dist, nn = est.kneighbors(dataset.X_test, return_dataframe_index=True)
    ndarrays_regression.check(dict(dist=dist, nn=nn))


@pytest.mark.uncollect_if(func=estimator_does_not_support_n_components)
@pytest.mark.parametrize(
    "estimator", ESTIMATOR_RESULTS.values(), ids=ESTIMATOR_RESULTS.keys()
)
@pytest.mark.parametrize("n_components", [None, 3], ids=["full", "reduced"])
@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("reference", [True, False], ids=["reference", "target"])
def test_predict(ndarrays_regression, estimator, n_components, weighted, reference):
    """Test that the ported estimators predict and score the correct values."""
    dataset = load_moscow_stjoes_test_data()

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
