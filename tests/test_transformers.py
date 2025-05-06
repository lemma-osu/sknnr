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


@pytest.mark.parametrize("x_type", ["array", "dataframe"])
def test_rfnode_transformer_assigns_correct_forest_types(x_type):
    """Test that the RFNodeTransformer returns the correct forest types."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=x_type == "dataframe")
    est = RFNodeTransformer().fit(X, y)
    assert all(v == "regression" for v in est.rf_type_dict_.values())
    assert all(isinstance(forest, RandomForestRegressor) for forest in est.rfs_)

    y_bool = y.astype("bool")
    est = RFNodeTransformer().fit(X, y_bool)
    assert all(v == "classification" for v in est.rf_type_dict_.values())
    assert all(isinstance(forest, RandomForestClassifier) for forest in est.rfs_)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("x_type", ["array", "series"])
def test_rfnode_transformer_handles_mixed_target(x_type):
    """Test that the RFNodeTransformer correctly upcasts mixed targets."""
    X, y = load_moscow_stjoes(return_X_y=True)
    y = y[:, 0].astype(object)
    if x_type == "series":
        y = pd.Series(y)
        y.iloc[-1] = "mixed"
    else:
        y[-1] = "mixed"
    est = RFNodeTransformer().fit(X, y)
    assert est.rf_type_dict_["0"] == "classification"
    assert isinstance(est.rfs_[0], RandomForestClassifier)
