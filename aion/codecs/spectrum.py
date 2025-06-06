import torch
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Real
from typing import Type

from aion.modalities import Spectrum
from aion.codecs.modules.convnext import ConvNextDecoder1d, ConvNextEncoder1d
from aion.codecs.modules.spectrum import LatentSpectralGrid
from aion.codecs.quantizers import LucidrainsLFQ, Quantizer, ScalarLinearQuantizer
from aion.codecs.base import Codec


class AutoencoderSpectrumCodec(Codec):
    """Meta-class for autoencoder codecs for spectra, does not actually contains a network."""

    def __init__(
        self,
        quantizer: Quantizer,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        normalization_quantizer: Quantizer,
        lambda_min: float = 3500.0,
        resolution: float = 0.8,
        num_pixels: int = 8704,
        latent_channels: int = 512,
        embedding_dim: int = 4,
        clip_ivar: float = 100,
        clip_flux: float | None = None,
        input_scaling: float = 0.2,
    ):
        super().__init__()
        self._quantizer = quantizer
        self.encoder = encoder
        self.decoder = decoder
        self.normalization_quantizer = normalization_quantizer
        self.latent_grid = LatentSpectralGrid(
            lambda_min=lambda_min, resolution=resolution, num_pixels=num_pixels
        )
        self.embedding_dim = embedding_dim
        self.clip_ivar = clip_ivar
        self.clip_flux = clip_flux
        self.input_scaling = input_scaling
        self.pre_quant_norm = torch.nn.LayerNorm(latent_channels)
        self.quant_conv = torch.nn.Conv1d(latent_channels, embedding_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embedding_dim, latent_channels, 1)

    @property
    def modality(self) -> Type[Spectrum]:
        return Spectrum

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _encode(self, x: Spectrum) -> Float[torch.Tensor, "b c t"]:
        # Extract fields from Spectrum instance
        flux = x.flux
        ivar = x.ivar
        mask = x.mask
        wavelength = x.wavelength

        # Robustify the model against NaN values in the input
        # And add optional cliping of extreme values
        spectrum = torch.nan_to_num(flux)
        if self.clip_flux is not None:
            spectrum = torch.clamp(spectrum, -self.clip_flux, self.clip_flux)
        ivar = torch.nan_to_num(ivar)
        if self.clip_ivar is not None:
            ivar = torch.clamp(ivar, 0, self.clip_ivar)
        istd = torch.sqrt(ivar)

        # Normalize input spectrum
        normalization = (spectrum * (1.0 - mask.float())).sum(dim=-1) / (
            torch.count_nonzero(~mask, dim=-1) + 1.0
        )

        normalization = torch.clamp(normalization, 0.1)

        # Compressing the range of this normalization factor
        normalization = torch.log10(normalization + 1.0)

        # Apply quantization to normalization factor
        normalization = self.normalization_quantizer.quantize(normalization)

        # Normalize the spectrum
        n = torch.clamp((10 ** normalization[..., None] - 1.0), 0.1)
        spectrum = (spectrum / n - 1.0) * self.input_scaling
        istd = (istd / n) * self.input_scaling

        # Project spectra on the latent grid
        spectrum = self.latent_grid.to_latent(spectrum, wavelength)
        istd = self.latent_grid.to_latent(istd, wavelength)

        # Apply additional range compression for good measure
        x = torch.arcsinh(torch.stack([spectrum, istd], dim=1))
        h = self.encoder(x)
        h = self.pre_quant_norm(h.moveaxis(1, -1)).moveaxis(-1, 1)
        h = self.quant_conv(h)
        return h, normalization

    def encode(self, x: Spectrum) -> Real[torch.Tensor, " b code"]:
        # Override to handle normalization token
        # First verify input type
        if not isinstance(x, self.modality):
            raise ValueError(
                f"Input type {type(x).__name__} does not match the modality of the codec {self.modality.__name__}"
            )

        # Get embedding using _encode
        embedding, normalization = self._encode(x)

        # Quantize embedding
        embedding = self.quantizer.encode(embedding)

        # Quantize normalization
        normalization = self.normalization_quantizer.encode(normalization)

        # Concatenate normalization token with embedding
        embedding = torch.cat([normalization[..., None], embedding], dim=-1)

        return embedding

    def decode(
        self,
        z: Real[torch.Tensor, " b code"],
        wavelength: Float[torch.Tensor, " b t"] | None = None,
    ) -> Spectrum:
        # Override to handle normalization token extraction
        # Extract the normalization token from the sequence
        norm_token, z = z[..., 0], z[..., 1:]

        normalization = self.normalization_quantizer.decode(norm_token)

        z = self.quantizer.decode(z)

        return self._decode(z, normalization=normalization, wavelength=wavelength)

    def _decode(
        self,
        z: Float[torch.Tensor, " b c l"],
        normalization: Float[torch.Tensor, " b"],
        wavelength: Float[torch.Tensor, " b t"] | None = None,
    ) -> Spectrum:
        h = self.post_quant_conv(z)
        spectra = self.decoder(h)

        if spectra.shape[1] == 1:  # just flux
            spectra = spectra.squeeze(1)
            mask = torch.ones_like(spectra) * -torch.inf
        elif spectra.shape[1] == 2:  # flux and mask
            spectra, mask = spectra.chunk(2, dim=1)
            spectra, mask = spectra.squeeze(1), mask.squeeze(1)
        else:
            raise ValueError("Invalid number of output channels, must be 1 or 2")

        # If the wavelength are provided, interpolate the spectrum on the observed grid
        if wavelength is not None:
            spectra = self.latent_grid.to_observed(spectra, wavelength)
            mask = self.latent_grid.to_observed(mask, wavelength)
        else:
            b = spectra.shape[0]
            wavelength = self.latent_grid.wavelength.reshape(1, -1).repeat(b, 1)

        # Decode the spectrum on the latent grid and apply normalization
        if normalization is not None:
            spectra = (spectra + 1.0) * torch.clamp(
                10 ** normalization[..., None] - 1.0, 0.1
            )

        # Round mask
        mask = torch.round(torch.sigmoid(mask)).bool().detach()

        # Return Spectrum instance
        return Spectrum(
            flux=spectra,
            ivar=torch.ones_like(spectra),  # We don't decode ivar, so set to ones
            mask=mask,
            wavelength=wavelength,
        )


class SpectrumCodec(AutoencoderSpectrumCodec, PyTorchModelHubMixin):
    """Spectrum codec based on convnext blocks."""

    def __init__(
        self,
        encoder_depths: tuple[int, ...] = (3, 3, 9, 3),
        encoder_dims: tuple[int, ...] = (96, 192, 384, 768),
        decoder_depths: tuple[int, ...] = (3, 3, 9, 3),
        decoder_dims: tuple[int, ...] = (384, 192, 96, 1),
        lambda_min: float = 3500.0,
        resolution: float = 0.8,
        num_pixels: int = 8704,
        latent_channels: int = 512,
        embedding_dim: int = 4,
        clip_ivar: float = 100,
        clip_flux: float | None = None,
        input_scaling: float = 0.2,
        normalization_range: tuple[float, float] = (-1, 5),
        codebook_size: int = 1024,
        dim: int = 10,
    ):
        assert encoder_dims[-1] == latent_channels, (
            "Last encoder dim must match latent_channels"
        )
        quantizer = LucidrainsLFQ(dim=dim, codebook_size=codebook_size)
        normalization_quantizer = ScalarLinearQuantizer(
            codebook_size=codebook_size, range=normalization_range
        )
        encoder = ConvNextEncoder1d(
            in_chans=2,
            depths=encoder_depths,
            dims=encoder_dims,
        )

        decoder = ConvNextDecoder1d(
            in_chans=latent_channels,
            depths=decoder_depths,
            dims=decoder_dims,
        )
        super().__init__(
            quantizer=quantizer,
            encoder=encoder,
            decoder=decoder,
            normalization_quantizer=normalization_quantizer,
            lambda_min=lambda_min,
            resolution=resolution,
            num_pixels=num_pixels,
            latent_channels=latent_channels,
            embedding_dim=embedding_dim,
            clip_ivar=clip_ivar,
            clip_flux=clip_flux,
            input_scaling=input_scaling,
        )
