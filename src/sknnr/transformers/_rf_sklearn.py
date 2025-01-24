from sklearn.ensemble import RandomForestRegressor


def sklearn_get_forest(X, y, n_tree, mt):
    """
    Train a random forest regression model in sklearn.
    """
    rf = RandomForestRegressor(
        n_estimators=n_tree, max_features=mt, random_state=42, min_samples_leaf=5
    )
    rf.fit(X, y)
    return rf


def sklearn_get_nodeset(rf, X):
    """
    Get the nodes associated with X of the random forest regression model in sklearn.
    """
    return rf.apply(X)
