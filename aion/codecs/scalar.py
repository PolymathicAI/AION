from typing import Type, Optional, Dict, Any

from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from torch import Tensor

from aion.codecs.quantizers import Quantizer, ScalarLinearQuantizer
from aion.codecs.quantizers.scalar import (
    ScalarLogReservoirQuantizer,
    ScalarReservoirQuantizer,
    MultiScalarCompressedReservoirQuantizer,
)
from aion.codecs.base import Codec
from aion.modalities import Scalar, ScalarModalities


class BaseScalarIdentityCodec(Codec, PyTorchModelHubMixin):
    """Codec for scalar quantities.

    A codec that embeds scalar quantities through an identity mapping. A
    quantizer is applied if specified.

    Args:
        modality_class: Type[ScalarModality]
            The modality class this codec is designed for.
        quantizer: Quantizer
            Optional quantizer for the scalar values.
    """

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    @property
    def modality(self) -> Type[Scalar]:
        return self._modality_class

    def _encode(self, x: Scalar) -> Float[Tensor, " b"]:
        return x.value

    def _decode(
        self, z: Float[Tensor, " b"], **metadata: Optional[Dict[str, Any]]
    ) -> Scalar:
        return self._modality_class(value=z)

    def load_state_dict(self, state_dict, strict=True):
        # This function is just because the scalar codecs were saved with 'quantizer' instead of '_quantizer'
        remapped_state_dict = {
            (
                k.replace("quantizer", "_quantizer", 1)
                if k.startswith("quantizer")
                else k
            ): v
            for k, v in state_dict.items()
        }
        return super().load_state_dict(remapped_state_dict, strict=strict)


class ScalarCodec(BaseScalarIdentityCodec):
    def __init__(
        self,
        modality: str,
        codebook_size: int,
        reservoir_size: int,
    ):
        super().__init__()
        self._modality_class = next(m for m in ScalarModalities if m.name == modality)
        self._quantizer = ScalarReservoirQuantizer(
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
        )


class LogScalarCodec(BaseScalarIdentityCodec):
    def __init__(
        self,
        modality: str,
        codebook_size: int,
        reservoir_size: int,
        min_log_value: float | None = -3,
    ):
        super().__init__()
        self._modality_class = next(m for m in ScalarModalities if m.name == modality)
        self._quantizer = ScalarLogReservoirQuantizer(
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
            min_log_value=min_log_value,
        )


class MultiScalarCodec(BaseScalarIdentityCodec):
    def __init__(
        self,
        modality: str,
        compression_fns: list[str],
        decompression_fns: list[str],
        codebook_size: int,
        reservoir_size: int,
        num_quantizers: int,
    ):
        super().__init__()
        self._modality_class = next(m for m in ScalarModalities if m.name == modality)
        self._quantizer = MultiScalarCompressedReservoirQuantizer(
            compression_fns=compression_fns,
            decompression_fns=decompression_fns,
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
            num_quantizers=num_quantizers,
        )


class GridScalarCodec(BaseScalarIdentityCodec):
    def __init__(self, modality: str, codebook_size: int):
        super().__init__()
        self._modality_class = next(m for m in ScalarModalities if m.name == modality)
        self._quantizer = ScalarLinearQuantizer(
            codebook_size=codebook_size,
            range=(0.0, 1.0),
        )
