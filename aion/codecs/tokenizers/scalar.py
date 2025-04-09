from aion.codecs.tokenizers.base import QuantizedCodec
from aion.codecs.quantizers import Quantizer

from jaxtyping import Float
import torch


class ScalarIdentityCodec(QuantizedCodec):
    """Codec for scalar quantities.

    A codec that embeds scalar quantities through an identity mapping.

    """

    def __init__(self, quantizer: Quantizer):
        super().__init__(quantizer)

    @property
    def modality(self):
        return "label"

    def _encode(self, x: Float[torch.Tensor, " b t"]) -> Float[torch.Tensor, " b t"]:
        return x

    def _decode(self, z: Float[torch.Tensor, " b c"]) -> Float[torch.Tensor, " b c"]:
        return z
