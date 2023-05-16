import pandas as pd
import pytest


class Dataset:
    def __init__(self, project, method, k=5):
        ref_distances_df = pd.read_csv(
            f"./tests/data/{method}_{project}_ref_distances_k{k}.csv"
        )
        ref_neighbors_df = pd.read_csv(
            f"./tests/data/{method}_{project}_ref_neighbors_k{k}.csv"
        )
        trg_distances_df = pd.read_csv(
            f"./tests/data/{method}_{project}_trg_distances_k{k}.csv"
        )
        trg_neighbors_df = pd.read_csv(
            f"./tests/data/{method}_{project}_trg_neighbors_k{k}.csv"
        )
        env_df = pd.read_csv(f"./tests/data/{project}_env.csv")
        spp_df = pd.read_csv(f"./tests/data/{project}_spp.csv")
        cols = [f"K{i+1}" for i in range(k)]

        self.ref_distances = ref_distances_df.loc[:, cols].values
        self.ref_neighbors = ref_neighbors_df.loc[:, cols].values
        self.trg_distances = trg_distances_df.loc[:, cols].values
        self.trg_neighbors = trg_neighbors_df.loc[:, cols].values

        self.X = env_df.iloc[:, 1:].values
        self.y = spp_df.iloc[:, 1:].values
        self.ids = env_df.iloc[:, 0].values


@pytest.fixture
def moscow_raw():
    return Dataset(project="moscow", method="raw", k=5)


@pytest.fixture
def moscow_euclidean():
    return Dataset(project="moscow", method="euclidean", k=5)


@pytest.fixture
def moscow_mahalanobis():
    return Dataset(project="moscow", method="mahalanobis", k=5)


@pytest.fixture
def moscow_msn():
    return Dataset(project="moscow", method="msn", k=5)


@pytest.fixture
def moscow_gnn():
    return Dataset(project="moscow", method="gnn", k=5)
