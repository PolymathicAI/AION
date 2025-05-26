"""Test the CodecManager class."""

from pathlib import Path

import pytest
import torch

from aion.codecs.manager import CodecManager, ModalityTypeError
from aion.modalities import (
    DESISpectrum,
    LegacySurveyFluxG,
    LegacySurveyImage,
    LegacySurveyShapeE1,
)


@pytest.fixture
def manager():
    """Create a CodecManager instance."""
    manager = CodecManager(device="cpu")
    yield manager
    manager._load_codec.cache_clear()


def test_encode_decode_image(manager: CodecManager, data_dir: Path):
    """Test encoding and decoding Image modality."""
    # Load test data
    input_batch_dict = torch.load(
        data_dir / "image_codec_input_batch.pt", weights_only=False
    )

    # Create Image modality
    image = LegacySurveyImage(
        flux=input_batch_dict["image"]["array"][:, 5:],
        bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
    )

    # Encode
    tokens = manager.encode(image)
    assert "tok_image" in tokens
    assert tokens["tok_image"].shape[0] == image.flux.shape[0]

    # Decode using modality type
    decoded_image = manager.decode(
        tokens, LegacySurveyImage, bands=["DES-G", "DES-R", "DES-I", "DES-Z"]
    )
    assert isinstance(decoded_image, LegacySurveyImage)
    assert decoded_image.flux.shape == image.flux.shape


def test_encode_decode_spectrum(manager: CodecManager, data_dir: Path):
    """Test encoding and decoding Spectrum modality."""
    # Load test data
    input_batch = torch.load(data_dir / "SPECTRUM_input_batch.pt", weights_only=False)[
        "spectrum"
    ]

    # Create Spectrum modality
    spectrum = DESISpectrum(
        flux=input_batch["flux"],
        ivar=input_batch["ivar"],
        mask=input_batch["mask"],
        wavelength=input_batch["lambda"],
    )

    # Encode
    tokens = manager.encode(spectrum)
    assert "tok_spectrum_desi" in tokens

    # Decode
    decoded_spectrum = manager.decode(tokens, DESISpectrum)
    assert isinstance(decoded_spectrum, DESISpectrum)
    assert decoded_spectrum.flux.shape[0] == spectrum.flux.shape[0]
    # Spectrum are returned with a fixed length
    assert decoded_spectrum.flux.shape[1] >= spectrum.flux.shape[1]


def test_codec_caching(manager: CodecManager):
    """Test that codecs are properly cached and reused."""
    # Create two modalities that use the same codec type
    flux_g1 = LegacySurveyFluxG(value=torch.randn(4, 1))
    flux_g2 = LegacySurveyFluxG(value=torch.randn(4, 1))

    # Encode both
    manager.encode(flux_g1)
    manager.encode(flux_g2)

    # Check that only one codec was loaded
    assert manager._load_codec.cache_info().hits == 1

    # Check that the same codec instance is used
    codec1 = manager._load_codec(LegacySurveyFluxG)
    codec2 = manager._load_codec(LegacySurveyFluxG)
    assert codec1 is codec2


def test_error_handling(manager: CodecManager):
    """Test error handling in CodecManager."""

    # Test with invalid modality type
    class InvalidModality:
        pass

    with pytest.raises(ModalityTypeError):
        manager._load_codec(InvalidModality)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_different_batch_sizes(manager: CodecManager, batch_size: int):
    """Test that CodecManager handles different batch sizes correctly."""
    # Create modalities with different batch sizes
    flux_g = LegacySurveyFluxG(value=torch.randn(batch_size, 1))
    shape_e1 = LegacySurveyShapeE1(value=torch.randn(batch_size, 1))

    # Encode
    tokens = manager.encode(flux_g, shape_e1)

    # Check batch sizes
    assert tokens["tok_flux_g"].shape[0] == batch_size
    assert tokens["tok_shape_e1"].shape[0] == batch_size

    # Decode and verify
    decoded_flux = manager.decode(tokens, LegacySurveyFluxG)
    decoded_shape = manager.decode(tokens, LegacySurveyShapeE1)

    assert decoded_flux.value.shape[0] == batch_size
    assert decoded_shape.value.shape[0] == batch_size
