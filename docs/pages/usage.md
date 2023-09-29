## Estimators

`sknnr` provides estimators that are fully compatible, drop-in replacements for `scikit-learn` estimators:

- [RawKNNRegressor](api/estimators/raw.md)
- [EuclideanKNNRegressor](api/estimators/euclidean.md)
- [MahalanobisKNNRegressor](api/estimators/mahalanobis.md)
- [GNNRegressor](api/estimators/gnn.md)
- [MSNRegressor](api/estimators/msn.md)

These estimators can be used [like any other `sklearn` regressor](https://scikit-learn.org/stable/getting_started.html#fitting-and-predicting-estimator-basics):

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

### Independent Scores and Predictions

When an independent test set is not available, the accuracy of a kNN regressor can be estimated by comparing each sample in the training set to its second-nearest neighbor, i.e. the closest point *excluding itself*. All `sknnr` estimators set `independent_prediction_` and `independent_score_` attributes when they are fit, which store the predictions and scores of this independent evaluation.

```python
print(est.independent_score_)
# 0.10243925752772305
```

### Retrieving Dataframe Indexes

In `sklearn`, the `KNeighborsRegressor.kneighbors` method can identify the array index of the nearest neighbor to a given sample. Estimators in `sknnr` offer an additional parameter `return_dataframe_index` that allows neighbor plots to be identified directly by their index.

```python
X, y = load_swo_ecoplot(return_X_y=True, as_frame=True)
est = est.fit(X, y)

# Find the distance and dataframe index of the nearest neighbors to the first plot
distances, neighbor_ids = est.kneighbors(X.iloc[:1], return_dataframe_index=True)

# Preview the nearest neighbors by their dataframe index
print(y.loc[neighbor_ids[0]])
```

|       |   ABAM_COV |   ABGRC_COV |   ABPRSH_COV |   ACMA3_COV |   ALRH2_COV |
|------:|-----------:|------------:|-------------:|------------:|------------:|
| 52481 |          0 |           0 |      39.3469 |           0 |           0 |
| 60089 |          0 |           0 |      22.1199 |           0 |           0 |
| 56253 |          0 |           0 |      22.8948 |           0 |           0 |

!!! warning
    An estimator must be fit with a `DataFrame` in order to use `return_dataframe_index=True`.

### Y-Fit Data

The [GNNRegressor](api/estimators/gnn.md) and [MSNRegressor](api/estimators/msn.md) estimators can be fit with `X` and `y` data, but they also accept an optional `y_fit` parameter. If provided, `y_fit` is used to fit the ordination transformer while `y` is used to fit the kNN regressor.

```python
from sknnr import GNNRegressor

est = GNNRegressor().fit(X, y, y_fit=y_fit)
```

### Dimensionality Reduction

The ordination transformers used by the [GNNRegressor](api/estimators/gnn.md) and [MSNRegressor](api/estimators/msn.md) estimators apply dimensionality reduction to label data. Dimensionality can be further reduced by specifying `n_components` when instantiating the estimator.

```python
est = GNNRegressor(n_components=3).fit(X, y)
```

!!! warning
    The maximum number of components depends on the input data and the estimator. Specifying `n_components` greater than the maximum number of components will raise an error.

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

|       |   ANNPRE |   ANNTMP |   AUGMAXT |   CONTPRE |   CVPRE |   DECMINT |   DIFTMP |   SMRTMP |   SMRTP |   ASPTR |     DEM |     PRR |   SLPPCT |   TPI450 |     TC1 |      TC2 |      TC3 |     NBR |
|------:|---------:|---------:|----------:|----------:|--------:|----------:|---------:|---------:|--------:|--------:|--------:|--------:|---------:|---------:|--------:|---------:|---------:|--------:|
| 52481 |  740     |  514.667 |   2315    |   517.667 | 8971.67 |  -583.111 |  2899.11 |  1136.11 | 212.222 | 197.667 | 1870.11 | 13196.7 |  48.3333 |  33.7778 | 218.778 |  68.5556 | -86.2222 | 343.556 |
| 52482 |  742     |  563.556 |   2354.33 |   502     | 9124.33 |  -543.556 |  2898.89 |  1179.44 | 221.111 | 190.222 | 1713.11 | 16355.8 |   5.4444 |   6.4444 | 210.222 |  60.3333 | -96.6667 | 261.667 |
| 52484 |  738.556 |  639.111 |   2468.89 |   545.889 | 8897.22 |  -479.111 |  2949    |  1266.22 | 236     | 194.556 | 1612.11 | 15132.6 |  15.5556 |  -1.2222 | 157     | 110.222  | -17.4444 | 721     |
| 52485 |  730.333 |  622.667 |   2405.33 |   555     | 8829.78 |  -481.222 |  2887.56 |  1244.22 | 234     | 196.444 | 1682.33 | 15146.7 |  19.8889 | -16.8889 | 152.556 |  86.1111 | -31.6667 | 597.111 |
| 52494 |  720     |  778.556 |   2678.11 |   658.556 | 8638    |  -386.667 |  3065.78 |  1396    | 262     | 191.778 | 1345.67 | 16672.1 |   2      |   0.4444 | 214.667 |  58.5556 | -88.1111 | 294.222 |

!!! note
    `pandas` must be installed to use `as_frame=True`.
