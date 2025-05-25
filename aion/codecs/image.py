import torch
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from torch import Tensor
from typing import Type, Optional, List

from aion.modalities import Image
from aion.codecs.modules.magvit import MagVitAE
from aion.codecs.modules.subsampler import SubsampledLinear
from aion.codecs.quantizers import FiniteScalarQuantizer, Quantizer
from aion.codecs.base import Codec
from aion.codecs.preprocessing.image import (
    ImagePadder,
    CenterCrop,
    RescaleToLegacySurvey,
    Clamp,
)
from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX


class AutoencoderImageCodec(Codec):
    """Meta-class for autoencoder codecs for images, does not actually contain a network."""

    def __init__(
        self,
        quantizer: Quantizer,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        hidden_dims: int = 64,
        embedding_dim: int = 5,
        multisurvey_projection_dims: int = 54,
        range_compression_factor: float = 0.01,
        mult_factor: float = 10.0,
    ):
        super().__init__()
        self._quantizer = quantizer
        self.range_compression_factor = range_compression_factor
        self.mult_factor = mult_factor
        self.encoder = encoder
        self.decoder = decoder

        # Preprocessing
        self.clamp = Clamp()
        self.center_crop = CenterCrop(crop_size=96)
        self.rescaler = RescaleToLegacySurvey()

        # Handle multi-survey projection
        self.image_padder = ImagePadder()
        self.subsample_in = SubsampledLinear(
            dim_in=self.image_padder.nbands,
            dim_out=multisurvey_projection_dims,
            subsample_in=True,
        )
        self.subsample_out = SubsampledLinear(
            dim_in=multisurvey_projection_dims,
            dim_out=self.image_padder.nbands,
            subsample_in=False,
        )
        # Go down to size of levels
        self.pre_quant_proj = torch.nn.Conv2d(
            hidden_dims, embedding_dim, kernel_size=1, stride=1, padding=0
        )

        # Go back to the original size
        self.post_quant_proj = torch.nn.Conv2d(
            embedding_dim, hidden_dims, kernel_size=1, stride=1, padding=0
        )

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    @property
    def modality(self) -> Type[Image]:
        return Image

    def _get_survey(self, bands: List[str]) -> str:
        survey = bands[0].split("-")[0]
        return survey

    def _range_compress(self, x: Tensor) -> Tensor:
        x = (
            torch.arcsinh(x / self.range_compression_factor)
            * self.range_compression_factor
        )
        x = x * self.mult_factor
        return x

    def _reverse_range_compress(self, x: Tensor) -> Tensor:
        x = x / self.mult_factor
        x = (
            torch.sinh(x / self.range_compression_factor)
            * self.range_compression_factor
        )
        return x

    def _encode(self, x: Image) -> Float[torch.Tensor, "b c w*h"]:
        flux_tensor = x.flux
        bands_in = x.bands

        processed_flux = self.center_crop(flux_tensor)
        processed_flux = self.clamp(processed_flux, bands_in)
        processed_flux = self.rescaler.forward(
            processed_flux, self._get_survey(bands_in)
        )
        processed_flux = self._range_compress(processed_flux)

        processed_flux, channel_mask = self.image_padder.forward(
            processed_flux, bands_in
        )
        processed_flux = self.subsample_in(processed_flux, channel_mask)

        h = self.encoder(processed_flux)
        h = self.pre_quant_proj(h)

        # Flatten the spatial dimensions
        h = h.reshape(h.shape[0], h.shape[1], -1)
        return h

    def _decode(
        self, z: Float[torch.Tensor, "b c w*h"], bands: Optional[List[str]] = None
    ) -> Image:
        # z is flattened, need to reshape
        batch_size, embedding_dim, n_tokens = z.shape
        spatial_size = int(n_tokens**0.5)
        z = z.reshape(batch_size, embedding_dim, spatial_size, spatial_size)

        h = self.post_quant_proj(z)
        decoded_flux_raw = self.decoder(h)

        full_dim_channel_mask = torch.ones(
            (z.shape[0], self.image_padder.nbands), device=z.device, dtype=torch.bool
        )
        decoded_flux_padded = self.subsample_out(
            decoded_flux_raw, full_dim_channel_mask
        )

        decoded_flux_compressed = self._reverse_range_compress(decoded_flux_padded)

        if bands is None:
            target_bands = list(BAND_TO_INDEX.keys())
        else:
            target_bands = bands

        final_flux = self.image_padder.backward(decoded_flux_compressed, target_bands)
        final_flux = self.rescaler.backward(final_flux, self._get_survey(target_bands))

        return Image(flux=final_flux, bands=target_bands)

    def decode(
        self, z: Float[Tensor, "b c"], bands: Optional[List[str]] = None
    ) -> Image:
        """
        Decodes the given latent tensor `z` back into an Image object.

        Args:
            z: The latent tensor to decode.
            bands (Optional[List[str]]): A list of band names to decode.
                                         If None or not provided, all default bands ('DES-G', 'DES-R', 'DES-I', 'DES-Z',
                                        'HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y')
                                         will be decoded.
        Returns:
            An Image object.
        """
        return super().decode(z, bands=bands)


class ImageCodec(AutoencoderImageCodec, PyTorchModelHubMixin):
    def __init__(
        self,
        quantizer_levels: List[int],
        hidden_dims: int = 512,
        multisurvey_projection_dims: int = 54,
        n_compressions: int = 2,
        num_consecutive: int = 4,
        embedding_dim: int = 5,
        range_compression_factor: float = 0.01,
        mult_factor: float = 10.0,
    ):
        """
        MagViT Autoencoder for images.

        Args:
            quantizer_levels: Levels for the FiniteScalarQuantizer.
            hidden_dims: Number of hidden dimensions in the network.
            n_compressions: Number of compressions in the network.
            num_consecutive: Number of consecutive residual layers per compression.
            embedding_dim: Dimension of the latent space.
            range_compression_factor: Range compression factor.
            mult_factor: Multiplication factor.
        """
        model = MagVitAE(
            n_bands=multisurvey_projection_dims,
            hidden_dims=hidden_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )
        quantizer = FiniteScalarQuantizer(levels=quantizer_levels)
        super().__init__(
            quantizer,
            model.encode,
            model.decode,
            hidden_dims,
            embedding_dim,
            multisurvey_projection_dims,
            range_compression_factor,
            mult_factor,
        )
        self.model = model
