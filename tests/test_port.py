from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.model_selection import train_test_split

from sklearn_knn import Raw, Euclidean, GNN


# def test_moscow_raw(moscow_raw):
#     clf = Raw(n_neighbors=5).fit(moscow_raw.X, moscow_raw.ids)
#     dist, _ = clf.kneighbors()
#     nn = clf.kneighbor_ids()

#     assert_array_equal(nn, moscow_raw.neighbors)
#     assert_array_almost_equal(dist, moscow_raw.distances, decimal=3)


# def test_moscow_euc(moscow_euc):
#     clf = Euclidean(n_neighbors=5).fit(moscow_euc.X, moscow_euc.ids)
#     dist, _ = clf.kneighbors()
#     nn = clf.kneighbor_ids()

#     assert_array_equal(nn, moscow_euc.neighbors)
#     assert_array_almost_equal(dist, moscow_euc.distances, decimal=3)


# def test_moscow_gnn(moscow_gnn):
#     clf = GNN(n_neighbors=5).fit(moscow_gnn.X, moscow_gnn.ids, spp=moscow_gnn.y)
#     dist, _ = clf.kneighbors()
#     nn = clf.kneighbor_ids()

#     assert_array_equal(nn, moscow_gnn.neighbors)
#     assert_array_almost_equal(dist, moscow_gnn.distances, decimal=3)


def test_moscow_raw(moscow_raw):
    X_train, X_test, y_train, _ = train_test_split(
        moscow_raw.X, moscow_raw.ids, train_size=0.8, shuffle=False
    )
    clf = Raw(n_neighbors=5).fit(X_train, y_train)

    dist, _ = clf.kneighbors()
    nn = clf.kneighbor_ids()

    assert_array_equal(nn, moscow_raw.ref_neighbors)
    assert_array_almost_equal(dist, moscow_raw.ref_distances, decimal=3)

    dist, _ = clf.kneighbors(X_test)
    nn = clf.kneighbor_ids(X_test)

    assert_array_equal(nn, moscow_raw.trg_neighbors)
    assert_array_almost_equal(dist, moscow_raw.trg_distances, decimal=3)


def test_moscow_euc(moscow_euc):
    X_train, X_test, y_train, _ = train_test_split(
        moscow_euc.X, moscow_euc.ids, train_size=0.8, shuffle=False
    )
    clf = Euclidean(n_neighbors=5).fit(X_train, y_train)

    dist, _ = clf.kneighbors()
    nn = clf.kneighbor_ids()

    assert_array_equal(nn, moscow_euc.ref_neighbors)
    assert_array_almost_equal(dist, moscow_euc.ref_distances, decimal=3)

    dist, _ = clf.kneighbors(X_test)
    nn = clf.kneighbor_ids(X_test)

    assert_array_equal(nn, moscow_euc.trg_neighbors)
    assert_array_almost_equal(dist, moscow_euc.trg_distances, decimal=3)


def test_moscow_gnn(moscow_gnn):
    X_train, X_test, y_train, _, y_spp, _ = train_test_split(
        moscow_gnn.X,
        moscow_gnn.ids,
        moscow_gnn.y,
        train_size=0.8,
        shuffle=False,
    )
    clf = GNN(n_neighbors=5).fit(X_train, y_train, spp=y_spp)

    dist, _ = clf.kneighbors()
    nn = clf.kneighbor_ids()

    assert_array_equal(nn, moscow_gnn.ref_neighbors)
    assert_array_almost_equal(dist, moscow_gnn.ref_distances, decimal=3)

    dist, _ = clf.kneighbors(X_test)
    nn = clf.kneighbor_ids(X_test)

    assert_array_equal(nn, moscow_gnn.trg_neighbors)
    assert_array_almost_equal(dist, moscow_gnn.trg_distances, decimal=3)
