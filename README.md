# scikit-learn-knn-regression

> [!WARNING]  
> This package is in active development!

`scikit-learn-knn-regression` (a.k.a. `sknnr`) is a package for running k-nearest neighbor imputation methods, including GNN (Ohmann & Gregory, 2002), MSN (Moeur & Stage 1995), and RFNN (Crookston & Finley, 2008), using estimators that are fully compatible with [`scikit-learn`](https://scikit-learn.org/stable/).


## Acknowledgements

`sknnr` was inspired by the [yaImpute](https://cran.r-project.org/web/packages/yaImpute/index.html) package for R (Crookston & Finley 2008). Thanks to Andrew Hudak for the inclusion of the [Moscow Mountain / St. Joes dataset](api/datasets/moscow_stjoes.md) (Hudak 2010), and Tom DeMeo for the inclusion of the [SWO Ecoplot dataset](api/datasets/swo_ecoplot.md) (Atzet et al., 1996). Development of this package was funded in part by an appointment to the United States Forest Service (USFS) Research Participation Program administered by the Oak Ridge Institute for Science and Education (ORISE) through an interagency agreement between the U.S. Department of Energy (DOE) and the U.S. Department of Agriculture (USDA).

## References

- Atzet, T, DE White, LA McCrimmon, PA Martinez, PR Fong, and VD Randall. 1996. Field guide to the forested plant associations of southwestern Oregon. USDA Forest Service. Pacific Northwest Region, Technical Paper R6-NR-ECOL-TP-17-96.
- Crookston, NL, Finley, AO. 2008. yaImpute: An R package for kNN imputation. Journal of Statistical Software, 23(10), 16. 
- Hudak, A.T. 2010. Field plot measures and predictive maps for "Nearest neighbor imputation of species-level, plot-scale forest structure attributes from LiDAR data". Fort Collins, CO: U.S. Department of Agriculture, Forest Service, Rocky Mountain Research Station. https://www.fs.usda.gov/rds/archive/Catalog/RDS-2010-0012.
- Moeur M, Stage AR. 1995. Most Similar Neighbor: An Improved Sampling Inference Procedure for Natural Resources Planning. Forest Science, 41(2), 337–359.
- Ohmann JL, Gregory MJ. 2002. Predictive Mapping of Forest Composition and Structure with Direct Gradient Analysis and Nearest Neighbor Imputation in Coastal Oregon, USA. Canadian Journal of Forest Research, 32, 725–741.
