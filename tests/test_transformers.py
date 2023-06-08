from typing import List, Type

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn import set_config
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import NotFittedError

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
def test_transformers_get_feature_names_out(transformer, moscow_euclidean):
    """Test that all transformers get the correct number of feature names out."""
    fit_transformer = transformer().fit(X=moscow_euclidean.X, y=moscow_euclidean.y)
    feature_names = fit_transformer.get_feature_names_out()
    X_transformed = fit_transformer.transform(X=moscow_euclidean.X)

    assert feature_names.shape == (X_transformed.shape[1],)


@pytest.mark.parametrize("config_type", ["global", "transformer"])
@pytest.mark.parametrize("output_mode", ["default", "pandas"])
@pytest.mark.parametrize("x_type", ["array", "dataframe"])
@pytest.mark.parametrize("transformer", get_transformer_classes())
def test_transformer_output_type_consistency(
    config_type, output_mode, x_type, transformer, moscow_euclidean
):
    """Test that output types are consistent with an sklearn transformer."""
    X, y = moscow_euclidean.X, moscow_euclidean.y
    X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    x = X if x_type == "array" else X_df

    transformer = transformer()
    ref_transformer = StandardScaler()

    if config_type == "global":
        set_config(transform_output=output_mode)

    else:
        transformer.set_output(transform=output_mode)
        ref_transformer.set_output(transform=output_mode)

    sknnr_type = type(transformer.fit_transform(x, y))
    ref_type = type(ref_transformer.fit_transform(x, y))

    assert sknnr_type is ref_type  # noqa: E721


@pytest.mark.parametrize("config_type", ["global", "transformer"])
@pytest.mark.parametrize("output_mode", ["default", "pandas"])
@pytest.mark.parametrize("x_type", ["array", "dataframe"])
@pytest.mark.parametrize("transformer", get_transformer_classes())
def test_transformer_feature_consistency(
    config_type, output_mode, x_type, transformer, moscow_euclidean
):
    """Test that feature names are consistent with an sklearn transformer."""
    X, y = moscow_euclidean.X, moscow_euclidean.y
    x = (
        X
        if x_type == "array"
        else pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    )
    transformer = transformer()
    ref_transformer = StandardScaler()

    if config_type == "global":
        set_config(transform_output=output_mode)

    else:
        transformer.set_output(transform=output_mode)
        ref_transformer.set_output(transform=output_mode)

    if hasattr(ref_transformer.fit(x, y), "feature_names_in_"):
        assert_array_equal(
            transformer.fit(x, y).feature_names_in_,
            ref_transformer.fit(x, y).feature_names_in_,
        )
    else:
        assert not hasattr(transformer.fit(x, y), "feature_names_in_")


@pytest.mark.parametrize("transformer", get_transformer_classes())
def test_transformers_raise_notfitted_transform(transformer, moscow_euclidean):
    """Attempting to call transform on an unfitted transformer should raise."""
    with pytest.raises(NotFittedError):
        transformer().transform(moscow_euclidean.X)
