from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor
from typing import Dict, Type, Optional, Any

from aion.modalities import ModalityType, Modality
from aion.codecs.quantizers import Quantizer


class Codec(ABC, torch.nn.Module):
    """Abstract definition of the Codec API.

    A Codec is responsible for transforming data of a specific modality into a
    continuous latent representation, which is then quantized into discrete tokens.
    It also provides the functionality to decode these tokens back into the
    original data space.
    """

    @property
    @abstractmethod
    def modality(self) -> Type[Modality]:
        """Returns the modality key that this codec can operate on."""
        raise NotImplementedError

    @abstractmethod
    def _encode(self, x: ModalityType) -> Float[Tensor, "b c n_tokens"]:
        """Function to be implemented by subclasses which
        takes a batch of input samples (as a ModalityType instance)
        and embeds it into a latent space, before any quantization.
        """
        raise NotImplementedError

    @abstractmethod
    def _decode(
        self, z: Float[Tensor, "b c n_tokens"], **metadata: Optional[Dict[str, Any]]
    ) -> ModalityType:
        """Function to be implemented by subclasses which
        takes a batch of latent space embeddings (after dequantization)
        and decodes it into the original input space as a ModalityType instance.

        Args:
            z: The batch of latent space embeddings after dequantization.
            **metadata: Optional keyword arguments containing metadata that might be
                necessary for the decoding process (e.g., original dimensions,
                specific modality parameters).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def quantizer(self) -> "Quantizer":
        """Returns the quantizer."""
        raise NotImplementedError

    def encode(self, x: ModalityType) -> Float[Tensor, "b n_tokens"]:
        """Encodes a given batch of samples into latent space.
        Encodes a batch of input samples into quantized discrete tokens.

        This involves first embedding the input into a continuous latent space
        using `_encode`, and then quantizing this embedding using the
        associated `quantizer`.

        Args:
            x: A batch of input samples (as a ModalityType instance).

        Returns:
            A tensor representing the quantized discrete tokens.
        """
        # Verify that the input type corresponds to the modality of the codec
        if not isinstance(x, self.modality):
            raise ValueError(
                f"Input type {type(x).__name__} does not match the modality of the codec {self.modality.__name__}"
            )
        embedding = self._encode(x)
        return self.quantizer.encode(embedding)

    def decode(
        self, z: Float[Tensor, "b n_tokens"], **metadata: Optional[Dict[str, Any]]
    ) -> ModalityType:
        """Decodes a batch of quantized discrete tokens back into the original data space.

        This involves first dequantizing the tokens using the associated `quantizer`,
        and then decoding the resulting continuous latent representation using `_decode`.

        Args:
            z: A tensor representing the quantized discrete tokens.
            **metadata: Optional keyword arguments containing metadata that might be
                necessary for the decoding process, passed to `_decode`.

        Returns:
            The decoded batch of samples as a ModalityType instance.
        """
        z = self.quantizer.decode(z)
        return self._decode(z, **metadata)
