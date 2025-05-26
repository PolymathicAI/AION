"""Codec Manager for AION.

Handles dynamic loading and management of codecs for different modalities.
"""

from typing import Dict, Optional, Type, Union

import torch

from aion.codecs.base import Codec
from aion.codecs.config import CODEC_CONFIG
from aion.modalities import Modality


class ModalityTypeError(TypeError):
    """Error raised when a modality type is not supported."""


class TokenKeyError(ValueError):
    """Error raised when a token key is not found in the tokens dictionary."""


class CodecManager:
    """Manager for loading and using codecs for different modalities."""

    def __init__(
        self, device: Union[str, torch.device] = "cpu", cache_dir: Optional[str] = None
    ):
        """Initialize the codec manager.

        Args:
            device: Device to load codecs on
            cache_dir: Optional cache directory for downloaded models
        """
        self.device = device
        self.cache_dir = cache_dir
        self._codecs: Dict[str, Codec] = {}  # Cache by repo_id to handle shared codecs
        self._modality_to_codec: Dict[
            Type[Modality], Codec
        ] = {}  # Map modality types to codecs

    def _get_codec_for_modality(self, modality_type: Type[Modality]) -> Codec:
        """Get or load the appropriate codec for a modality."""

        # Check if codec is already loaded for this modality type
        if modality_type in self._modality_to_codec:
            return self._modality_to_codec[modality_type]

        # Load codec based on modality type
        codec = self._load_codec(modality_type)
        self._modality_to_codec[modality_type] = codec
        return codec

    def _load_codec(self, modality_type: Type[Modality]) -> Codec:
        """Load a codec for the given modality type."""
        # Look up configuration in CODEC_CONFIG
        if modality_type not in CODEC_CONFIG:
            # Try base class if specific modality not found
            if (
                hasattr(modality_type, "__base__")
                and modality_type.__base__ in CODEC_CONFIG
            ):
                config = CODEC_CONFIG[modality_type.__base__]
            else:
                raise ModalityTypeError(
                    f"No codec configuration found for modality type: {modality_type.__name__}"
                )
        else:
            config = CODEC_CONFIG[modality_type]

        repo_id = config["repo_id"]

        # Check if this codec has already been loaded (shared codec case)
        if repo_id in self._codecs:
            return self._codecs[repo_id]

        codec_class = config["class"]

        # Load from HuggingFace
        codec = codec_class.from_pretrained(repo_id, cache_dir=self.cache_dir)
        codec = codec.to(self.device).eval()

        # Cache by repo_id to handle shared codecs
        self._codecs[repo_id] = codec

        return codec

    @torch.no_grad()
    def encode(self, *modalities: Modality) -> Dict[str, torch.Tensor]:
        """Encode multiple modalities.

        Args:
            *modalities: Variable number of modality instances to encode

        Returns:
            Dictionary mapping token keys to encoded tensors
        """
        tokens = {}

        for modality in modalities:
            # Get the appropriate codec
            codec = self._get_codec_for_modality(type(modality))

            # Tokenize the modality
            tokenized = codec.encode(modality)

            # Get the token key for this modality
            if hasattr(modality, "token_key"):
                token_key = modality.token_key
            else:
                raise ModalityTypeError(
                    f"Modality {type(modality).__name__} does not have a token_key attribute"
                )

            tokens[token_key] = tokenized

        return tokens

    @torch.no_grad()
    def decode(
        self,
        tokens: Dict[str, torch.Tensor],
        modality_type: Type[Modality],
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
        if not hasattr(modality_type, "token_key"):
            raise ModalityTypeError(
                f"Modality type {modality_type} does not have a token_key attribute"
            )

        token_key = modality_type.token_key
        if token_key not in tokens:
            raise TokenKeyError(
                f"Token key '{token_key}' for modality {modality_type} not found in tokens dictionary"
            )

        # Get the appropriate codec
        codec = self._get_codec_for_modality(modality_type)

        # Decode using the codec with any provided metadata
        decoded_modality = codec.decode(tokens[token_key], **metadata)

        # Cast decoded modality to the correct type
        decoded_modality = modality_type(**decoded_modality.model_dump())

        return decoded_modality
