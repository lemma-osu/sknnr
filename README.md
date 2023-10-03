# sknnr

> ⚠️ **WARNING: sknnr is in active development!** ⚠️

## What is sknnr?

`sknnr` is a package for running k-nearest neighbor (kNN) imputation methods using estimators that are fully compatible with [`scikit-learn`](https://scikit-learn.org/stable/). Notably, common methods such as most similar neighbor (MSN, Moeur & Stage 1995), gradient nearest neighbor (GNN, Ohmann & Gregory, 2002), and random forest nearest neighbors (RFNN, Crookston & Finley, 2008) are included in this package.

## Why the name "sknnr"?

`sknnr` is an acronym of its main three components:

1. **"s"** is for `scikit-learn`. All estimators in this package derive from the `sklearn.BaseEstimator` class and comply with the requirements associated with [developing custom estimators](https://scikit-learn.org/stable/developers/develop.html).
2. **"knn"** is for k-nearest neighbors. All estimators use the _k_ >= 1 samples that are nearest in feature space to create their prediction. Each estimator in this package defines that feature space in a different way which often leads to different neighbors chosen for the prediction.
3. **"r"** is for regression. Estimators in this package are run in [regression mode](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html). For nearest neighbor imputation, this is simply an (optionally-weighted) average of its _k_ neighbors. When _k_ is set to 1, this effectively behaves as in classification mode. All estimators support multioutput prediction so that multiple features can be predicted with the same estimator.

## Quick-Start

1. Follow the [installation guide](https://sknnr.readthedocs.io/installation).
2. Import any `sknnr` estimator, like [MSNRegressor](https://sknnr.readthedocs.io/api/estimators/msn), as a drop-in replacement for a `scikit-learn` regressor.
```python
from sknnr import MSNRegressor

est = MSNRegressor()
```
3. Load a custom dataset like [SWO Ecoplot](https://sknnr.readthedocs.io/api/datasets/swo_ecoplot) (or bring your own).
```python
from sknnr.datasets import load_swo_ecoplot

X, y = load_swo_ecoplot(return_X_y=True, as_frame=True)
```
4. Train, predict, and score [as usual](https://scikit-learn.org/stable/getting_started.html#fitting-and-predicting-estimator-basics).
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

est = est.fit(X_train, y_train)
est.score(X_test, y_test)
```
5. Check out the additional features like [independent scoring](https://sknnr.readthedocs.io/usage/#independent-scores-and-predictions), [dataframe indexing](https://sknnr.readthedocs.io/usage/#retrieving-dataframe-indexes), and [dimensionality reduction](https://sknnr.readthedocs.io/usage/#dimensionality-reduction).
```python
# Evaluate the model using the second-nearest neighbor in the test set
print(est.fit(X, y).independent_score_)

# Get the dataframe index of the nearest neighbor to each plot
print(est.kneighbors(return_dataframe_index=True, return_distance=False))

# Apply dimensionality reduction using CCorA ordination
MSNRegressor(n_components=3).fit(X_train, y_train)
```

## History and Inspiration
`sknnr` was heavily inspired by (and endeavors to implement functionality of) the [yaImpute](https://cran.r-project.org/web/packages/yaImpute/index.html) package for R (Crookston & Finley 2008).  As Crookston and Finley (2008) note in their abstract,
> Although nearest neighbor imputation is used in a host of disciplines, the methods implemented in the yaImpute package are tailored to imputation-based forest attribute estimation and mapping ... [there is] a growing interest in nearest neighbor imputation methods for spatially explicit forest inventory, and a need within this research community for software that facilitates comparison among different nearest neighbor search algorithms and subsequent imputation techniques.

Indeed, many regional (e.g. [LEMMA](https://lemmadownload.forestry.oregonstate.edu/)) and national (e.g. [BIGMAP](https://storymaps.arcgis.com/stories/c710684b98f54452804e8960d37905b2), [TreeMap](https://www.firelab.org/project/treemap-tree-level-model-forests-united-states)) projects use nearest-neighbor methods to 
estimate and map forest attributes across time and space.

To that end, `sknnr` ports and expands the functionality present in `yaImpute` into a Python package that helps facilitate intercomparison between k-nearest neighbor methods (and other built-in estimators from `scikit-learn`) using an API which is familiar to `scikit-learn` users.

## Acknowledgements

Thanks to Andrew Hudak (USDA Forest Service Rocky Mountain Research Station) for the inclusion of the [Moscow Mountain / St. Joes dataset](https://sknnr.readthedocs.io/api/datasets/moscow_stjoes) (Hudak 2010), and the USDA Forest Service Region 6 Ecology Team for the inclusion of the [SWO Ecoplot dataset](https://sknnr.readthedocs.io/api/datasets/swo_ecoplot) (Atzet et al., 1996). Development of this package was funded by:

* an appointment to the United States Forest Service (USFS) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Department of Agriculture (USDA).
* a joint venture agreement between USFS Pacific Northwest Research Station and Oregon State University (agreement 19-JV-11261959-064).
* a cost-reimbursable agreement between USFS Region 6 and Oregon State University (agreeement 21-CR-11062756-046).

## References

- Atzet, T, DE White, LA McCrimmon, PA Martinez, PR Fong, and VD Randall. 1996. Field guide to the forested plant associations of southwestern Oregon. USDA Forest Service. Pacific Northwest Region, Technical Paper R6-NR-ECOL-TP-17-96.
- Crookston, NL, Finley, AO. 2008. yaImpute: An R package for kNN imputation. Journal of Statistical Software, 23(10), 16.
- Hudak, A.T. 2010. Field plot measures and predictive maps for "Nearest neighbor imputation of species-level, plot-scale forest structure attributes from LiDAR data". Fort Collins, CO: U.S. Department of Agriculture, Forest Service, Rocky Mountain Research Station. https://www.fs.usda.gov/rds/archive/Catalog/RDS-2010-0012.
- Moeur M, Stage AR. 1995. Most Similar Neighbor: An Improved Sampling Inference Procedure for Natural Resources Planning. Forest Science, 41(2), 337–359.
- Ohmann JL, Gregory MJ. 2002. Predictive Mapping of Forest Composition and Structure with Direct Gradient Analysis and Nearest Neighbor Imputation in Coastal Oregon, USA. Canadian Journal of Forest Research, 32, 725–741.
