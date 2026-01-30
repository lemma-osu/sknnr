## Estimators

`sknnr` provides six estimators that are fully compatible, drop-in replacements for `scikit-learn` estimators:

- [RawKNNRegressor](api/estimators/raw.md)
- [EuclideanKNNRegressor](api/estimators/euclidean.md)
- [MahalanobisKNNRegressor](api/estimators/mahalanobis.md)
- [GNNRegressor](api/estimators/gnn.md)
- [MSNRegressor](api/estimators/msn.md)
- [RFNNRegressor](api/estimators/rfnn.md)

These estimators can be used like any other `sklearn` regressor (or [classifier](#regression-and-classification))[^sklearn-docs].

[^sklearn-docs]: Check out the [sklearn docs](https://scikit-learn.org/stable/getting_started.html#fitting-and-predicting-estimator-basics) for a refresher on estimator basics.

```python
from sknnr import EuclideanKNNRegressor
from sknnr.datasets import load_swo_ecoplot
from sklearn.model_selection import train_test_split

X, y = load_swo_ecoplot(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
est = EuclideanKNNRegressor(n_neighbors=3).fit(X_train, y_train)

print(est.score(X_test, y_test))
# 0.11496218649569434
```

In addition to their core functionality of fitting, predicting, and scoring, `sknnr` estimators offer a number of other features, detailed below.

### Regression and Classification

The estimators in `sknnr` are all initialized with an optional parameter `n_neighbors` that determines how many plots a target plot's attributes will be predicted from. When `n_neighbors` > 1, a plot's attributes are calculated as optionally-weighted averages of each of its _k_ nearest neighbors. Predicted values can fall anywhere between the observed plot values, making this "regression mode" suitable for continuous attributes (e.g. basal area). To maintain categorical attributes (e.g. dominant species type), the estimators can be run in "classification mode" with `n_neighbors` = 1, where each attribute is imputed directly from its nearest neighbor. To predict a combination of continuous and categorical attributes, it's possible to use two estimators and concatenate their predictions manually.

### Independent Scores and Predictions

When an independent test set is not available, the accuracy of a kNN regressor can be estimated by comparing each sample in the training set to its second-nearest neighbor, i.e. the closest point _excluding itself_. All `sknnr` estimators set `independent_prediction_` and `independent_score_` attributes when they are fit, which store the predictions and scores of this independent evaluation.

```python
print(est.independent_score_)
# 0.10243925752772305
```

### Deterministic Neighbor Ordering

`scikit-learn`'s `KNeighborsRegressor` [warns](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) that:

> in case of multiple neighbors being at the same distance, the result will depend on the order of the samples in the training data.

In `sknnr`, we allow the user to enforce strict ordering of neighbors with deterministic tie-breaking when calling `kneighbors` by using the `use_deterministic_ordering` parameter. When this value is `True`, neighbors are sorted using the following logical order:

1. **Scaled and rounded distances**: Neighbors are first sorted by their distances rounded to `DISTANCE_PRECISION_DECIMALS` decimal places (currently set to 10). Some floating point operations in distance determination (notably `numpy.dot`) can introduce very small numerical differences across platforms, which is effectively handled by this rounding.
2. **Difference between query point row index and neighbors indexes**: If two or more neighbors have identical rounded distances, they are further sorted by the absolute difference between their row index in the training data and the row index of the query point. This ensures that when a sample is its own nearest neighbor, it will always be selected first.
3. **Neighbor index**: If two or more neighbors are still tied based on the two above criteria, they are finally sorted by their row index in the training data.

As an example, consider the following training data with three samples:

```python
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3]
])
y = np.array([10, 20, 30])
est = KNeighborsRegressor(n_neighbors=2).fit(X, y)

print(est.kneighbors(X, return_distance=False))
# [[0 2]
#  [1 0]
#  [0 2]] - Not returning itself as first neighbor
```

Using `sknnr`'s `RawKNNRegressor` with deterministic ordering:

```python
from sknnr import RawKNNRegressor
est = RawKNNRegressor(n_neighbors=2).fit(X, y)
print(est.kneighbors(X, return_distance=False, use_deterministic_ordering=True))
# [[0 2]
#  [1 0]
#  [2 0]] - Returning itself as first neighbor
```

The `use_deterministic_ordering` parameter defaults to `True`, but can revert to `scikit-learn`'s default behavior when calling `kneighbors`:

```python
distances, neighbors = est.kneighbors(
    X_test,
    use_deterministic_ordering=False
)
```

### Retrieving Dataframe Indexes

In `sklearn`, the `KNeighborsRegressor.kneighbors` method can identify the array index of the nearest neighbor to a given sample. Estimators in `sknnr` offer an additional parameter `return_dataframe_index` that allows neighbor samples to be identified directly by their index.

```python
X, y = load_swo_ecoplot(return_X_y=True, as_frame=True)
est = est.fit(X, y)

# Find the distance and dataframe index of the nearest neighbors to the first plot
distances, neighbor_ids = est.kneighbors(X.iloc[:1], return_dataframe_index=True)

# Preview the nearest neighbors by their dataframe index
print(y.loc[neighbor_ids[0]])
```

|       | ABAM_COV | ABGRC_COV | ABPRSH_COV | ACMA3_COV | ALRH2_COV |
| ----: | -------: | --------: | ---------: | --------: | --------: |
| 52481 |        0 |         0 |    39.3469 |         0 |         0 |
| 60089 |        0 |         0 |    22.1199 |         0 |         0 |
| 56253 |        0 |         0 |    22.8948 |         0 |         0 |

!!! warning
    An estimator must be fit with a `DataFrame` in order to use `return_dataframe_index=True`.

!!! tip
    In forestry applications, users typically store a unique inventory plot identification number as the index in the dataframe.

### Y-Fit Data

The [GNNRegressor](api/estimators/gnn.md), [MSNRegressor](api/estimators/msn.md), and [RFNNRegressor](api/estimators/rfnn.md) estimators can be fit with `X` and `y` data, but they also accept an optional `y_fit` parameter. If provided, `y_fit` is used to fit the ordination transformer while `y` is used to fit the kNN regressor.

In forest attribute estimation, the underlying ordination transformations for two of these estimators (CCA for GNN and CCorA for MSN) typically use a matrix of species abundances or presence/absence information to relate the species data to environmental covariates, but often the user wants predictions based not on these features, but rather attributes that describe forest structure (e.g. biomass) or composition (e.g. species richness). In this case, the species matrix would be specified as `y_fit` and the stand attributes would be specified as `y`.

For RFNN, the `y_fit` parameter can be used to specify the attributes for which individual random forests will be created (one forest per feature). As with GNN and MSN, the `y` parameter can then be used to specify the attributes that will be predicted by the nearest neighbors.

```python
from sknnr import GNNRegressor

est = GNNRegressor().fit(X, y, y_fit=y_fit)
```

### Dimensionality Reduction

The ordination transformers used by the [GNNRegressor](api/estimators/gnn.md) and [MSNRegressor](api/estimators/msn.md) estimators apply dimensionality reduction by creating components that are linear combinations of the features in the `X` data. For both transformers, components that explain more variation present in the `y` (or `y_fit`) matrix are ordered first. Users can further reduce the number of components that are used to determine nearest neighbors by specifying `n_components` when instantiating the estimator.

```python
est = GNNRegressor(n_components=3).fit(X, y)
```

!!! warning
    The maximum number of components depends on the input data and the estimator. Specifying `n_components` greater than the maximum number of components will raise an error.

### RFNN Distance Metric

For all estimators other than [RFNNRegressor](api/estimators/rfnn.md), the distance metric used to determine nearest neighbors is the Euclidean distance between samples in the transformed space. RFNN, on the other hand, first builds a random forest for each feature in the `y` (or `y_fit`) matrix and then captures the node IDs (_not_ values) for each sample on every forest and tree. The distance between samples is calculated using [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance), which captures the number of node IDs that are different between the target and reference samples and then divided by the total number of nodes. Therefore, a target and reference sample that share _all_ node IDs would have a distance of 0, whereas a target and reference sample that share _no_ node IDs would have a distance of 1.

### Custom Transformers

Most estimators in `sknnr` work by applying specialized transformers like [CCA](api/transformers/cca.md) and [CCorA](api/transformers/ccora.md) to the input data. These transformers can be used independently of the estimators, like any other `sklearn` transformer.

```python
from sknnr.transformers import CCATransformer

cca = CCATransformer(n_components=3)
print(cca.fit_transform(X, y))
```

`sknnr` currently provides the following transformers:

- [StandardScalerWithDOF](api/transformers/standardscalerwithdof.md)
- [MahalanobisTransformer](api/transformers/mahalanobis.md)
- [CCATransformer](api/transformers/cca.md)
- [CCorATransformer](api/transformers/ccora.md)
- [RFNodeTransformer](api/transformers/rfnode.md)

## Datasets

`sknnr` estimators can be used for any multi-output regression problem, but they excel at predicting forest attributes. The `sknnr.datasets` module contains a number of test datasets with plot-based forest measurements and environmental attributes.

```python
from sknnr.datasets import load_swo_ecoplot, load_moscow_stjoes
```

### Dataset Format

Like in `sklearn`, datasets in `sknnr` can be loaded in a variety of formats, including as a `dict`-like [`Dataset` object](api/datasets/dataset.md):

```python
dataset = load_swo_ecoplot()
print(dataset)
# Dataset(n=3005, features=18, targets=25)
```

...as an X, y `tuple` of Numpy arrays:

```python
X, y = load_swo_ecoplot(return_X_y=True)
print(X.shape, y.shape)
# (3005, 18) (3005, 25)
```

...or as `tuple` of Pandas dataframes:

```python
X_df, y_df = load_swo_ecoplot(return_X_y=True, as_frame=True)
print(X_df.head())
```

|       |  ANNPRE |  ANNTMP | AUGMAXT | CONTPRE |   CVPRE |  DECMINT |  DIFTMP |  SMRTMP |   SMRTP |   ASPTR |     DEM |     PRR |  SLPPCT |   TPI450 |     TC1 |     TC2 |      TC3 |     NBR |
| ----: | ------: | ------: | ------: | ------: | ------: | -------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | -------: | ------: | ------: | -------: | ------: |
| 52481 |     740 | 514.667 |    2315 | 517.667 | 8971.67 | -583.111 | 2899.11 | 1136.11 | 212.222 | 197.667 | 1870.11 | 13196.7 | 48.3333 |  33.7778 | 218.778 | 68.5556 | -86.2222 | 343.556 |
| 52482 |     742 | 563.556 | 2354.33 |     502 | 9124.33 | -543.556 | 2898.89 | 1179.44 | 221.111 | 190.222 | 1713.11 | 16355.8 |  5.4444 |   6.4444 | 210.222 | 60.3333 | -96.6667 | 261.667 |
| 52484 | 738.556 | 639.111 | 2468.89 | 545.889 | 8897.22 | -479.111 |    2949 | 1266.22 |     236 | 194.556 | 1612.11 | 15132.6 | 15.5556 |  -1.2222 |     157 | 110.222 | -17.4444 |     721 |
| 52485 | 730.333 | 622.667 | 2405.33 |     555 | 8829.78 | -481.222 | 2887.56 | 1244.22 |     234 | 196.444 | 1682.33 | 15146.7 | 19.8889 | -16.8889 | 152.556 | 86.1111 | -31.6667 | 597.111 |
| 52494 |     720 | 778.556 | 2678.11 | 658.556 |    8638 | -386.667 | 3065.78 |    1396 |     262 | 191.778 | 1345.67 | 16672.1 |       2 |   0.4444 | 214.667 | 58.5556 | -88.1111 | 294.222 |

!!! note
    `pandas` must be installed to use `as_frame=True`.
