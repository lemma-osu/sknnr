import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError

from sknnr.datasets import load_moscow_stjoes
from sknnr.transformers import (
    CCATransformer,
    CCorATransformer,
    MahalanobisTransformer,
    RFNodeTransformer,
    StandardScalerWithDOF,
)

TEST_TRANSFORMERS = [
    StandardScalerWithDOF,
    MahalanobisTransformer,
    CCATransformer,
    CCorATransformer,
    RFNodeTransformer,
]

TEST_ORDINATION_TRANSFORMERS = [
    CCATransformer,
    CCorATransformer,
]

TEST_TREE_TRANSFORMERS = [
    RFNodeTransformer,
]

# Mapping of tree node transformers to their corresponding sklearn forest types
TREE_TRANSFORMER_FOREST_TYPES = {
    RFNodeTransformer: {
        "regression": RandomForestRegressor,
        "classification": RandomForestClassifier,
    },
}


def get_transformer_xfail_checks(transformer) -> dict[str, str]:
    """
    Return tests that are expected to fail with explanations.

    These are mostly due to sklearn using test data that our estimators aren't
    compatible with, e.g. 1D labels.

    Requires sklearn >= 1.6.
    """
    xfail_checks = {}

    if isinstance(transformer, CCATransformer):
        # These checks fail due to input data constraints for the CCA ordination that
        # aren't followed by the sklearn checks.
        one_d_checks = [
            "check_estimators_dtypes",
            "check_dtype_object",
            "check_estimators_fit_returns_self",
            "check_pipeline_consistency",
            "check_estimators_overwrite_params",
            "check_fit_score_takes_y",
            "check_estimators_pickle",
            "check_transformer_data_not_an_array",
            "check_transformer_general",
            "check_transformer_preserve_dtypes",
            "check_methods_sample_order_invariance",
            "check_methods_subset_invariance",
            "check_dict_unchanged",
            "check_dont_overwrite_parameters",
            "check_fit_idempotent",
            "check_fit_check_is_fitted",
            "check_n_features_in",
            "check_fit2d_predict1d",
            "check_fit2d_1sample",
            "check_estimators_nan_inf",
            "check_requires_y_none",
            "check_readonly_memmap_input",
            "check_n_features_in_after_fitting",
            "check_f_contiguous_array_estimator",
            "check_positive_only_tag_during_fit",
        ]

        xfail_checks.update(
            {check: "CCA requires 2D y arrays." for check in one_d_checks}
        )

    return xfail_checks


@parametrize_with_checks(
    [cls() for cls in TEST_TRANSFORMERS],
    expected_failed_checks=get_transformer_xfail_checks,
)
def test_sklearn_transformer_checks(estimator, check):
    check(estimator)


@pytest.mark.parametrize("transformer", TEST_TRANSFORMERS)
def test_transformers_get_feature_names_out(transformer):
    """Test that all transformers get the correct number of feature names out."""
    X, y = load_moscow_stjoes(return_X_y=True)
    fit_transformer = transformer().fit(X=X, y=y)
    feature_names = fit_transformer.get_feature_names_out()
    X_transformed = fit_transformer.transform(X=X)

    assert feature_names.shape == (X_transformed.shape[1],)


@pytest.mark.parametrize("config_type", ["global", "transformer"])
@pytest.mark.parametrize("output_mode", ["default", "pandas"])
@pytest.mark.parametrize("x_type", ["array", "dataframe"])
@pytest.mark.parametrize("transformer", TEST_TRANSFORMERS)
def test_transformer_output_type_consistency(
    config_type, output_mode, x_type, transformer
):
    """Test that output types are consistent with an sklearn transformer."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=x_type == "dataframe")

    transformer = transformer()
    ref_transformer = StandardScaler()

    if config_type == "global":
        global_config = {"transform_output": output_mode}
    else:
        global_config = {}
        transformer.set_output(transform=output_mode)
        ref_transformer.set_output(transform=output_mode)

    with config_context(**global_config):
        sknnr_type = type(transformer.fit_transform(X, y))
        ref_type = type(ref_transformer.fit_transform(X, y))

    assert sknnr_type is ref_type  # noqa: E721


@pytest.mark.parametrize("config_type", ["global", "transformer"])
@pytest.mark.parametrize("output_mode", ["default", "pandas"])
@pytest.mark.parametrize("x_type", ["array", "dataframe"])
@pytest.mark.parametrize("transformer", TEST_TRANSFORMERS)
def test_transformer_feature_consistency(config_type, output_mode, x_type, transformer):
    """Test that feature names are consistent with an sklearn transformer."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=x_type == "dataframe")
    transformer = transformer()
    ref_transformer = StandardScaler()

    if config_type == "global":
        global_config = {"transform_output": output_mode}
    else:
        global_config = {}
        transformer.set_output(transform=output_mode)
        ref_transformer.set_output(transform=output_mode)

    with config_context(**global_config):
        if hasattr(ref_transformer.fit(X, y), "feature_names_in_"):
            assert_array_equal(
                transformer.fit(X, y).feature_names_in_,
                ref_transformer.fit(X, y).feature_names_in_,
            )
        else:
            assert not hasattr(transformer.fit(X, y), "feature_names_in_")


@pytest.mark.parametrize("transformer", TEST_TRANSFORMERS)
def test_transformers_raise_notfitted_transform(transformer):
    """Attempting to call transform on an unfitted transformer should raise."""
    X, y = load_moscow_stjoes(return_X_y=True)
    with pytest.raises(NotFittedError):
        transformer().transform(X)


@pytest.mark.parametrize("transformer", TEST_ORDINATION_TRANSFORMERS)
@pytest.mark.parametrize("n_components", [None, 0, 5])
def test_transformers_n_components(transformer, n_components):
    """Test that n_components is handled correctly.

    Note: The value 5 was chosen because it was one component less
    than the minimum number of components across TEST_ORDINATION_TRANSFORMERS
    and should work across transformers."""
    X, y = load_moscow_stjoes(return_X_y=True)
    t = transformer(n_components=n_components).fit(X, y)
    if n_components is not None:
        assert t.n_components_ == n_components
    assert t.transform(X).shape[1] == t.n_components_


@pytest.mark.parametrize("transformer", TEST_ORDINATION_TRANSFORMERS)
@pytest.mark.parametrize("n_components", [-1, 1000])
def test_transformers_raise_out_of_range_n_components(transformer, n_components):
    """Attempting to call fit with an out of range value of n_components
    should raise."""
    X, y = load_moscow_stjoes(return_X_y=True)
    with pytest.raises(
        ValueError, match=r"n_components=-?\d+ must be between 0 and \d+"
    ):
        transformer(n_components=n_components).fit(X, y)


@pytest.mark.parametrize("transformer", TEST_TREE_TRANSFORMERS)
def test_treenode_transformer_assigns_correct_forest_types(transformer):
    """Test that the TreeNodeTransformer returns the correct forest types."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=False)

    transformer_type_dict = TREE_TRANSFORMER_FOREST_TYPES[transformer]
    clf_est_type = transformer_type_dict["classification"]
    reg_est_type = transformer_type_dict["regression"]

    est = transformer().fit(X, y)
    assert all(v == "regression" for v in est.estimator_type_dict_.values())
    assert all(isinstance(forest, reg_est_type) for forest in est.estimators_)

    y_bool = y.astype("bool")
    est = transformer().fit(X, y_bool)
    assert all(v == "classification" for v in est.estimator_type_dict_.values())
    assert all(isinstance(forest, clf_est_type) for forest in est.estimators_)


@pytest.mark.parametrize("transformer", TEST_TREE_TRANSFORMERS)
@pytest.mark.parametrize("y_wrapper", [pd.Series, np.asarray])
@pytest.mark.parametrize("nan_like_value", [np.nan, None, pd.NA])
def test_treenode_transformer_raises_on_nan_like_target(
    transformer, y_wrapper, nan_like_value
):
    """Test that the TreeNodeTransformer raises on targets with NaN-like elements."""
    X, y = load_moscow_stjoes(return_X_y=True)
    y = y[:, 0].astype(object)
    y[0] = nan_like_value
    y = y_wrapper(y, dtype=object)
    with pytest.raises(ValueError, match=r"Target \S+ has NaN-like elements"):
        _ = transformer().fit(X, y)


@pytest.mark.parametrize("transformer", TEST_TREE_TRANSFORMERS)
@pytest.mark.parametrize("y_wrapper", [pd.Series, np.asarray])
def test_treenode_transformer_raises_on_mixed_target(transformer, y_wrapper):
    """
    Test that the TreeNodeTransformer raises on targets with mixed
    string/non-string data that cannot safely be promoted to a common type.
    """
    X, y = load_moscow_stjoes(return_X_y=True)
    y = y[:, 0].astype(object)
    y[-1] = "mixed"
    y = y_wrapper(y, dtype=object)
    with pytest.raises(ValueError, match=r"Target \S+ has non-string types"):
        _ = transformer().fit(X, y)


@pytest.mark.parametrize("criterion_reg", ["absolute_error"])
@pytest.mark.parametrize("criterion_clf", ["entropy"])
@pytest.mark.parametrize("max_features_reg", ["sqrt", "log2"])
@pytest.mark.parametrize("max_features_clf", ["log2", 1.0])
@pytest.mark.parametrize("class_weight_clf", ["balanced_subsample"])
def test_rfnode_transformer_non_default_parameterization(
    criterion_reg,
    criterion_clf,
    max_features_reg,
    max_features_clf,
    class_weight_clf,
):
    """
    Test that RFNodeTransformer correctly passes specialized parameters
    to its regression and classification forests.
    """
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)

    # Create a boolean target for classification
    y["ABGR_PA"] = y["ABGR_BA"] > 0.0

    # Fit with the specialized non-default parameters
    est = RFNodeTransformer(
        criterion_reg=criterion_reg,
        criterion_clf=criterion_clf,
        max_features_reg=max_features_reg,
        max_features_clf=max_features_clf,
        class_weight_clf=class_weight_clf,
    ).fit(X, y)

    # Check that both regression and classification forests are present
    assert "classification" in est.estimator_type_dict_.values()
    assert "regression" in est.estimator_type_dict_.values()

    # Confirm that the specialized parameters are set on the correct forests
    for rf in est.estimators_:
        if isinstance(rf, RandomForestClassifier):
            assert rf.get_params()["criterion"] == criterion_clf
            assert rf.get_params()["max_features"] == max_features_clf
            assert rf.get_params()["class_weight"] == class_weight_clf
        else:
            assert rf.get_params()["criterion"] == criterion_reg
            assert rf.get_params()["max_features"] == max_features_reg


@pytest.mark.parametrize("forest_weights", ["uniform", [0.5, 1.5], (1.0, 2.0)])
def test_rfnode_transformer_handles_forest_weights(forest_weights):
    """Test that RFNodeTransformer handles forest weights correctly."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    num_weights = 1 if forest_weights == "uniform" else len(forest_weights)
    y = y.iloc[:, :num_weights]

    est = RFNodeTransformer(forest_weights=forest_weights).fit(X, y)

    if isinstance(forest_weights, str) and forest_weights == "uniform":
        expected_weights = np.full(
            (est.n_estimators, num_weights), 1.0, dtype="float64"
        )
    else:
        expected_weights = (
            np.array(forest_weights)
            * np.ones((est.n_estimators, num_weights), dtype="float64")
        ).T.flatten()

    assert hasattr(est, "tree_weights_")
    assert est.tree_weights_.shape == (est.n_total_trees_,)
    assert np.allclose(est.tree_weights_, expected_weights)


def test_rfnode_transformer_raises_on_invalid_forest_weights():
    """Test that RFNodeTransformer raises on invalid forest weights."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=True)
    y = y.iloc[:, :2]  # Use two targets

    with pytest.raises(ValueError, match=r"Expected `forest_weights` to have length 2"):
        RFNodeTransformer(forest_weights=[0.5]).fit(X, y)
