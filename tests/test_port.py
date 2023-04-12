from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd

from sklearn_knn import Raw, Euclidean, GNN

import pytest


class Dataset:
    def __init__(self, project, method, k=5):
        self.project = project
        self.method = method
        self.k = k
        
        self.distances = pd.read_csv(f"./tests/data/{method}_{project}_ref_distances_k{k}.csv")
        self.neighbors = pd.read_csv(f"./tests/data/{method}_{project}_ref_neighbors_k{k}.csv")
        self.env_df = pd.read_csv(f"./tests/data/{project}_env.csv")
        self.spp_df = pd.read_csv(f"./tests/data/{project}_spp.csv")

        cols = [f"K{i+1}" for i in range(k)]

        self.distances = self.distances.loc[:, cols].values
        self.neighbors = self.neighbors.loc[:, cols].values
        self.X = self.env_df.iloc[:, 1:].values
        self.Y = self.spp_df.iloc[:, 1:].values
        self.ids = self.env_df.iloc[:, 0].values
        self.spp = self.spp_df.iloc[:, 1:].values


@pytest.fixture
def moscow_raw():
    return Dataset(project="moscow", method="raw", k=5)


@pytest.fixture
def moscow_euc():
    return Dataset(project="moscow", method="euc", k=5)


@pytest.fixture
def moscow_gnn():
    return Dataset(project="moscow", method="gnn", k=5)


def test_moscow_raw(moscow_raw):
    clf = Raw(n_neighbors=5).fit(moscow_raw.X, moscow_raw.ids)
    dist, _ = clf.kneighbors()
    nn = clf.kneighbor_ids()

    assert_array_equal(nn, moscow_raw.neighbors)
    # assert_array_almost_equal(dist, moscow_raw.distances)


def test_moscow_euc(moscow_euc):
    clf = Euclidean(n_neighbors=5).fit(moscow_euc.X, moscow_euc.ids)
    dist, _ = clf.kneighbors()
    nn = clf.kneighbor_ids()

    assert_array_equal(nn, moscow_euc.neighbors)
    # assert_array_almost_equal(dist, moscow_euc.distances)


def test_moscow_gnn(moscow_gnn):
    clf = GNN(n_neighbors=5).fit(moscow_gnn.X, moscow_gnn.ids, cca_params={"spp": moscow_gnn.Y})
    dist, _ = clf.kneighbors()
    nn = clf.kneighbor_ids()

    assert_array_equal(nn, moscow_gnn.neighbors)
    # assert_array_almost_equal(dist, moscow_gnn.distances)
