import numpy as np
import pandas as pd

from sklearn_knn import Raw, Euclidean, GNN


def get_check_dist_fn(project, method, ref_or_trg):
    return f"./tests/data/{method}_{project}_{ref_or_trg}_distances_k5.csv"


def get_check_nn_fn(project, method, ref_or_trg):
    return f"./tests/data/{method}_{project}_{ref_or_trg}_neighbors_k5.csv"


def compare_array_results(calculated, check_fn, k, approx=True):
    check_df = pd.read_csv(check_fn)
    cols = [f"K{i+1}" for i in range(k)]
    check_arr = check_df.loc[:, cols].values
    return (
        np.allclose(calculated, check_arr)
        if approx
        else np.all(calculated == check_arr)
    )


def get_X_ids(project):
    env_df = pd.read_csv(f"./tests/data/{project}_env.csv")
    return env_df.iloc[:, 1:].values, env_df.iloc[:, 0].values


def get_Y(project):
    spp_df = pd.read_csv(f"./tests/data/{project}_spp.csv")
    return spp_df.iloc[:, 1:].values


def compare_results(clf, project, method, k):
    dist, _ = clf.kneighbors()
    nn = clf.kneighbor_ids()
    check_dist_fn = get_check_dist_fn(project, method, "ref")
    check_nn_fn = get_check_nn_fn(project, method, "ref")
    match_dists = compare_array_results(dist, check_dist_fn, k, approx=True)
    match_nns = compare_array_results(nn, check_nn_fn, k, approx=False)
    # return match_dists and match_nns
    return match_nns


def test_moscow_raw():
    X, ids = get_X_ids("moscow")
    clf = Raw(n_neighbors=5).fit(X, ids)
    assert compare_results(clf, "moscow", "raw", 5)


def test_moscow_euc():
    X, ids = get_X_ids("moscow")
    clf = Euclidean(n_neighbors=5).fit(X, ids)
    assert compare_results(clf, "moscow", "euc", 5)


def test_moscow_gnn():
    X, ids = get_X_ids("moscow")
    Y = get_Y("moscow")
    clf = GNN(n_neighbors=5).fit(X, ids, transform__spp=Y)
    assert compare_results(clf, "moscow", "gnn", 5)
