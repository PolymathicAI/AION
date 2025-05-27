"""Codec Manager for AION.

Handles dynamic loading and management of codecs for different modalities.
"""

from dataclasses import asdict
from functools import lru_cache

import torch

from aion.codecs.base import Codec
from aion.codecs.config import CODEC_CONFIG, CodecType
from aion.modalities import BaseModality, Modality


class ModalityTypeError(TypeError):
    """Error raised when a modality type is not supported."""


class TokenKeyError(ValueError):
    """Error raised when a token key is not found in the tokens dictionary."""


class CodecManager:
    """Manager for loading and using codecs for different modalities."""

    def __init__(self, device: str | torch.device = "cpu"):
        """Initialize the codec manager.

        Args:
            device: Device to load codecs on
            cache_dir: Optional cache directory for downloaded models
        """
        self.device = device

    @staticmethod
    @lru_cache
    def _load_codec_from_hf(codec_class: CodecType, hf_codec_repo_id: str) -> Codec:
        """Load a codec from HuggingFace.
        Although HF download is already cached,
        the method is cached to avoid reloading the same codec.

        Args:
            codec_class: The class of the codec to load
            hf_codec_repo_id: The HuggingFace repository ID of the codec

        Returns:
            The loaded codec
        """
        codec = codec_class.from_pretrained(hf_codec_repo_id)
        codec = codec.eval()
        return codec

    @lru_cache
    def _load_codec(self, modality_type: type[BaseModality]) -> Codec:
        """Load a codec for the given modality type."""
        # Look up configuration in CODEC_CONFIG
        if modality_type in CODEC_CONFIG:
            config = CODEC_CONFIG[modality_type]
        elif (
            hasattr(modality_type, "__base__")
            and modality_type.__base__ in CODEC_CONFIG
        ):
            config = CODEC_CONFIG[modality_type.__base__]
        else:
            raise ModalityTypeError(
                f"No codec configuration found for modality type: {modality_type.__name__}"
            )

        codec_class = config.codec_class
        hf_codec_repo_id = config.repo_id
        codec = self._load_codec_from_hf(codec_class, hf_codec_repo_id)

        return codec

    @torch.no_grad()
    def encode(self, *modalities: Modality) -> dict[str, torch.Tensor]:
        """Encode multiple modalities.

        Args:
            *modalities: Variable number of modality instances to encode

        Returns:
            Dictionary mapping token keys to encoded tensors
        """
        tokens = {}

        for modality in modalities:
            if not isinstance(modality, Modality):
                raise ModalityTypeError(
                    f"Modality {type(modality).__name__} does not have a token_key attribute"
                )
            # Get the appropriate codec
            codec = self._load_codec(type(modality))
            codec = codec.to(self.device)

            # Tokenize the modality
            tokenized = codec.encode(modality)

            tokens[modality.token_key] = tokenized

        return tokens

    @torch.no_grad()
    def decode(
        self,
        tokens: dict[str, torch.Tensor],
        modality_type: type[Modality],
        **metadata,
    ) -> Modality:
        """Decode tokens back to a modality.

        Args:
            tokens: Dictionary mapping token keys to tokenized tensors
            modality_type: The modality type (e.g., DESISpectrum) to decode into
            **metadata: Additional metadata required by the specific codec
                       (e.g., wavelength for spectra, bands for images)

        Returns:
            Decoded modality instance
        """
        if not issubclass(modality_type, Modality):
            raise ModalityTypeError(
                f"Modality type {modality_type} does not have a token_key attribute"
            )

        token_key = modality_type.token_key
        if token_key not in tokens:
            raise TokenKeyError(
                f"Token key '{token_key}' for modality {modality_type} not found in tokens dictionary"
            )

        # Get the appropriate codec
        codec = self._load_codec(modality_type)
        codec = codec.to(self.device)

        # Decode using the codec with any provided metadata
        decoded_modality = codec.decode(tokens[token_key], **metadata)

        # Cast decoded modality to the correct type
        decoded_modality = modality_type(**asdict(decoded_modality))

        return decoded_modality
