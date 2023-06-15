from typing import List, Type

import pytest
from numpy.testing import assert_array_equal
from sklearn import set_config
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import NotFittedError

from sknnr.datasets import load_moscow_stjoes
from sknnr.transformers import (
    CCATransformer,
    CCorATransformer,
    MahalanobisTransformer,
    StandardScalerWithDOF,
)


def get_transformer_classes() -> List[Type[TransformerMixin]]:
    """
    Return classes of all supported transformers.
    """
    return [
        StandardScalerWithDOF,
        MahalanobisTransformer,
        CCATransformer,
        CCorATransformer,
    ]


@pytest.mark.parametrize("transformer", get_transformer_classes())
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
@pytest.mark.parametrize("transformer", get_transformer_classes())
def test_transformer_output_type_consistency(
    config_type, output_mode, x_type, transformer
):
    """Test that output types are consistent with an sklearn transformer."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=x_type == "dataframe")

    transformer = transformer()
    ref_transformer = StandardScaler()

    if config_type == "global":
        set_config(transform_output=output_mode)

    else:
        transformer.set_output(transform=output_mode)
        ref_transformer.set_output(transform=output_mode)

    sknnr_type = type(transformer.fit_transform(X, y))
    ref_type = type(ref_transformer.fit_transform(X, y))

    assert sknnr_type is ref_type  # noqa: E721


@pytest.mark.parametrize("config_type", ["global", "transformer"])
@pytest.mark.parametrize("output_mode", ["default", "pandas"])
@pytest.mark.parametrize("x_type", ["array", "dataframe"])
@pytest.mark.parametrize("transformer", get_transformer_classes())
def test_transformer_feature_consistency(config_type, output_mode, x_type, transformer):
    """Test that feature names are consistent with an sklearn transformer."""
    X, y = load_moscow_stjoes(return_X_y=True, as_frame=x_type == "dataframe")
    transformer = transformer()
    ref_transformer = StandardScaler()

    if config_type == "global":
        set_config(transform_output=output_mode)

    else:
        transformer.set_output(transform=output_mode)
        ref_transformer.set_output(transform=output_mode)

    if hasattr(ref_transformer.fit(X, y), "feature_names_in_"):
        assert_array_equal(
            transformer.fit(X, y).feature_names_in_,
            ref_transformer.fit(X, y).feature_names_in_,
        )
    else:
        assert not hasattr(transformer.fit(X, y), "feature_names_in_")


@pytest.mark.parametrize("transformer", get_transformer_classes())
def test_transformers_raise_notfitted_transform(transformer):
    """Attempting to call transform on an unfitted transformer should raise."""
    X, y = load_moscow_stjoes(return_X_y=True)
    with pytest.raises(NotFittedError):
        transformer().transform(X)
