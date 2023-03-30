import sys

import pandas as pd
import numpy as np
from sklearn_knn import Raw, Euclidean


def get_check_dist_fn(project, method, ref_or_trg):
    return f"./tests/data/{method}_{project}_{ref_or_trg}_distances_k5.csv"


def get_check_nn_fn(project, method, ref_or_trg):
    return f"./tests/data/{method}_{project}_{ref_or_trg}_neighbors_k5.csv"


def compare(calculated, check_fn, k, approx=True):
    check_df = pd.read_csv(check_fn)
    cols = [f"K{i+1}" for i in range(k)]
    check_arr = check_df.loc[:, cols].values

    # # Finding differences in arrays
    # idx = np.where(~np.isclose(calculated, check_arr))
    # print(idx)
    # print(calculated[idx[0]][:10])
    # print(check_arr[idx[0]][:10])

    return (
        np.allclose(calculated, check_arr)
        if approx
        else np.all(calculated == check_arr)
    )


METHOD = {
    "raw": Raw,
    "euc": Euclidean,
}


def train_model(project, method, k):
    env_df = pd.read_csv(f"./tests/data/{project}_env.csv")
    X = env_df.iloc[:, 1:].values
    y = env_df.iloc[:, 0].values

    # X_train, y_train = X[:50], y[:50]
    # X_test, y_test = X[50:], y[50:]
    clf = METHOD[method](n_neighbors=k)
    clf.fit(X, y)
    clf.kneighbor_ids()
    return clf


def compare_results(project, method, k):
    clf = train_model(project, method, k)
    dist, _ = clf.kneighbors()
    nn = clf.kneighbor_ids()
    assert compare(dist, get_check_dist_fn(project, method, "ref"), k)
    assert compare(nn, get_check_nn_fn(project, method, "ref"), k)


if __name__ == "__main__":
    project, method = sys.argv[1:]
    compare_results(project, method, 5)

# Compare target data
# dist_trg, nn_trg = clf.kneighbors(X_test)
# assert compare(dist_trg, f"./tests/data/{prefix}_iris_trg_distances_k5.csv")
# assert compare(
#     y_train[nn_trg], f"./tests/data/{prefix}_iris_trg_neighbors_k5.csv", approx=False
# )
