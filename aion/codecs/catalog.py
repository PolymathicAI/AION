from collections import OrderedDict
from typing import Type, Optional, Dict

from huggingface_hub import PyTorchModelHubMixin
import torch
from jaxtyping import Float
from torch import Tensor

from aion.codecs.base import Codec
from aion.codecs.quantizers import Quantizer
from aion.codecs.quantizers.scalar import (
    ComposedScalarQuantizer,
    IdentityQuantizer,
    ScalarReservoirQuantizer,
)
from aion.modalities import LegacySurveyCatalog


class CatalogCodec(Codec, PyTorchModelHubMixin):
    """Codec for catalog quantities.

    A codec that embeds catalog quantities through an identity mapping. A
    quantizer is applied if specified.
    """

    def __init__(
        self,
        mask_value: int = 9999,
    ):
        super().__init__()
        self._modality = LegacySurveyCatalog
        catalog_keys = ["X", "Y", "SHAPE_E1", "SHAPE_E2", "SHAPE_R"]
        quantizers = [
            IdentityQuantizer(96),
            IdentityQuantizer(96),
            ScalarReservoirQuantizer(1024, 100000),
            ScalarReservoirQuantizer(1024, 100000),
            ScalarReservoirQuantizer(1024, 100000),
        ]
        self.mask_value = mask_value
        self._catalog_keys = catalog_keys
        assert len(catalog_keys) == len(quantizers), (
            "Number of catalog keys and quantizers must match"
        )
        _quantizer = OrderedDict()
        for key, quantizer in zip(catalog_keys, quantizers):
            _quantizer[key] = quantizer
        self._quantizer = ComposedScalarQuantizer(_quantizer)

    @property
    def modality(self) -> Type[LegacySurveyCatalog]:
        return self._modality

    @property
    def quantizer(self) -> Optional[Quantizer]:
        return self._quantizer

    def _encode(self, x: LegacySurveyCatalog) -> Dict[str, Tensor]:
        encoded = OrderedDict()
        for key in self._catalog_keys:
            catalog_value = getattr(x, key)
            mask = catalog_value != self.mask_value
            catalog_value = catalog_value[mask]
            encoded[key] = catalog_value
        encoded["mask"] = mask
        return encoded

    def encode(self, x: LegacySurveyCatalog) -> Float[Tensor, "b c1 *code_shape"]:
        """Encodes a given batch of samples into latent space."""
        embedding = self._encode(x)
        _encoded = self.quantizer.encode(
            embedding
        )  # (b, C), where b is the number of non-masked samples

        mask = embedding["mask"]
        # B: batch size, L: sequence length (20) for each catalog key
        B, L = mask.shape
        C = len(self._catalog_keys)
        encoded = self.mask_value * torch.ones(
            B, L, C, dtype=_encoded.dtype, device=_encoded.device
        )
        encoded[mask] = _encoded
        encoded = encoded.reshape(B, -1)
        return encoded

    def _decode(self, z: Dict[str, Tensor]) -> LegacySurveyCatalog:
        return LegacySurveyCatalog(**z)

    def decode(self, z: Float[Tensor, "b c1 *code_shape"]) -> LegacySurveyCatalog:
        B, LC = z.shape
        C = len(self._catalog_keys)
        L = LC // C
        z = z[:, : C * L]  # Truncate the z if it is longer than the expected length
        z = z.reshape(B * L, C)
        if self._quantizer is not None:
            z = self.quantizer.decode(z)
        for key in self._catalog_keys:
            z[key] = z[key].reshape(B, L)
        return self._decode(z)
