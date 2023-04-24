from typing import List

import pytest

# from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError

from sklearn_knn import GNN, Euclidean, Raw
from sklearn_knn._base import IDNeighborsClassifier


def get_kneighbor_estimator_instances() -> List[IDNeighborsClassifier]:
    """
    Return instances of all supported IDNeighborsClassifier estimators.
    """
    return [
        Raw(),
        Euclidean(),
        GNN(),
    ]


# Note: This will run all the sklearn estimator checks. It's going to take quite a bit
# of work to get these all passing, and it's possible we just won't be able to do it
# while maintaining all the features we need.
# @parametrize_with_checks(get_kneighbor_estimator_instances())
# def test_sklearn_compatibile_estimators(estimator, check):
#     check(estimator)


@pytest.mark.parametrize("estimator", get_kneighbor_estimator_instances())
def test_estimators_raise_notfitted_kneighbors(estimator, moscow_euc):
    """Attempting to call kneighbors on an unfitted estimator should raise."""
    with pytest.raises(NotFittedError):
        estimator.kneighbors(moscow_euc.X)


@pytest.mark.parametrize("estimator", get_kneighbor_estimator_instances())
def test_estimators_raise_notfitted_kneighbor_ids(estimator, moscow_euc):
    """Attempting to call kneighbor_ids on an unfitted estimator should raise."""
    with pytest.raises(NotFittedError):
        estimator.kneighbor_ids(moscow_euc.X)


@pytest.mark.parametrize("estimator", get_kneighbor_estimator_instances())
def test_estimators_raise_notfitted_predict(estimator, moscow_euc):
    """Attempting to call predict on an unfitted estimator should raise."""
    with pytest.raises(NotFittedError):
        estimator.predict(moscow_euc.X)
