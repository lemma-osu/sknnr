import pytest

from sknnr.transformers import (
    CCATransformer,
    CCorATransformer,
    MahalanobisTransformer,
    StandardScalerWithDOF,
)


def get_transformer_instances():
    """
    Return instances of all supported transformers.
    """
    return [
        StandardScalerWithDOF(),
        MahalanobisTransformer(),
        CCATransformer(),
        CCorATransformer(),
    ]


@pytest.mark.parametrize("transformer", get_transformer_instances())
def test_transformers_get_feature_names_out(transformer, moscow_euclidean):
    """Test that all transformers get the correct number of feature names out."""
    fit_transformer = transformer.fit(X=moscow_euclidean.X, y=moscow_euclidean.y)
    feature_names = fit_transformer.get_feature_names_out()
    X_transformed = fit_transformer.transform(X=moscow_euclidean.X)

    assert feature_names.shape == (X_transformed.shape[1],)


@pytest.mark.parametrize("transformer", get_transformer_instances())
def test_transformers_have_setoutput(transformer):
    """Test that all transformers can use set_output."""
    assert transformer.set_output() is not None
