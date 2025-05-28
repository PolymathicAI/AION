import pytest
import torch

from aion.posttraining.models import LinearProbing, CrossAttentionProbing
from aion.codecs.manager import CodecManager
from aion.modalities import LegacySurveyFluxG


@pytest.fixture(scope="module")
def tokens() -> dict[str, torch.Tensor]:
    codec_manager = CodecManager()

    batch_size = 8
    flux_g = LegacySurveyFluxG(value=torch.randn(batch_size, 1))

    # Encode
    tokens = codec_manager.encode(flux_g)

    return tokens


def test_linear_model(tokens: dict[str, torch.Tensor], aion_model_path: str):
    dim_out = 8

    model = LinearProbing(dim_out=dim_out, aion_model_path=aion_model_path)
    with torch.no_grad():
        output = model(tokens)

    batch_size = tokens["tok_flux_g"].shape[0]
    assert output.shape == (batch_size, dim_out)


def test_cross_attention_model(tokens: dict[str, torch.Tensor], aion_model_path: str):
    dim_out = 8
    num_heads = 2

    model = CrossAttentionProbing(dim_out, num_heads, aion_model_path)

    with torch.no_grad():
        output = model(tokens)

    batch_size = tokens["tok_flux_g"].shape[0]
    assert output.shape == (batch_size, dim_out)
