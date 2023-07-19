import pytest
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import NotFittedError

from sknnr.datasets import load_moscow_stjoes
from sknnr.transformers import (
    CCATransformer,
    CCorATransformer,
    MahalanobisTransformer,
    StandardScalerWithDOF,
)

TEST_TRANSFORMERS = [
    StandardScalerWithDOF,
    MahalanobisTransformer,
    CCATransformer,
    CCorATransformer,
]

TEST_ORDINATION_TRANSFORMERS = [
    CCATransformer,
    CCorATransformer,
]


@pytest.mark.xfail(reason="Incrementally fixing.")
@parametrize_with_checks([cls() for cls in TEST_TRANSFORMERS])
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
