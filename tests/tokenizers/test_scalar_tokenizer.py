import pytest
import torch

from aion.codecs.quantizers import IdentityQuantizer
from aion.codecs.tokenizers.scalar import ScalarIdentityCodec


@pytest.mark.parametrize("codebook_size", [10, 20])
@pytest.mark.parametrize("embedding_dim", [1, 4])
def test_scalar_identity_codec(codebook_size, embedding_dim):
    quantizer = IdentityQuantizer(codebook_size=codebook_size)
    codec = ScalarIdentityCodec(quantizer=quantizer)
    x = torch.randint(0, codebook_size, (64, embedding_dim))
    z = codec.encode(x)
    assert z.shape == x.shape
    assert torch.allclose(z, x)
