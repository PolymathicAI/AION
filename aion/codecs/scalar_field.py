from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Type
from jaxtyping import Float
from torch import Tensor

from huggingface_hub import PyTorchModelHubMixin

from .base import Codec
from .quantizers import Quantizer, FiniteScalarQuantizer
from .modules.convblocks import Encoder2d, Decoder2d
from .modules.ema import ModelEmaV2
from aion.modalities import LegacySurveySegmentationMap
from .preprocessing.image import CenterCrop


def _deep_get(dictionary, path, default=None):
    return reduce(lambda d, key: d[key], path.split("."), dictionary)


class AutoencoderScalarFieldCodec(Codec):
    """Abstract class for autoencoding scalar field codecs."""

    def __init__(
        # ------------------------------------------------------------------------------
        self,
        # Code dimensions --------------------------------------------------------------
        encoder_output_dim: int,
        decoder_input_dim: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        # Quantisation -----------------------------------------------------------------
        quantizer: Optional[Quantizer] = None,
        # VAE operation ----------------------------------------------------------------
        variational: bool = False,
        # Model outputs ----------------------------------------------------------------
        output_activation: Optional[Callable] = torch.sigmoid,
        output_activation_extension: Optional[float] = None,
        # Loss calculation -------------------------------------------------------------
        reconstruction_loss: Callable = F.mse_loss,
        quantisation_loss_weight: float = 1.0,
        # Loss optimisation ------------------------------------------------------------
        lr: float = 1e-3,
        lr_warmup: Optional[int] = None,
        begin_cosine_annealing: Optional[int] = None,
        lr_cosine_period: Optional[int] = None,
        # Model weights EMA ------------------------------------------------------------
        ema_model_weights: bool = False,
        ema_decay: float = 0.9999,
        ema_update_freq: int = 1,
        # ------------------------------------------------------------------------------
    ):
        super().__init__()

        # Code dimensions --------------------------------------------------------------

        self.encoder_output_dim = encoder_output_dim
        decoder_input_dim = decoder_input_dim or encoder_output_dim
        self.decoder_input_dim = decoder_input_dim
        embedding_dim = embedding_dim or encoder_output_dim
        self.embedding_dim = embedding_dim

        # VAE operation ---------------------------------------------------------------

        self.variational = variational

        # Preprocessing ----------------------------------------------------------------

        self.center_crop = CenterCrop(crop_size=96)

        # Quantisation -----------------------------------------------------------------

        # Pre/post quantisation projections
        encode_proj_dim = 2 * embedding_dim if variational else embedding_dim
        self.encode_proj = nn.Conv2d(encoder_output_dim, encode_proj_dim, 1)
        self.decode_proj = nn.Conv2d(embedding_dim, decoder_input_dim, 1)

        # Quantiser
        self._quantizer = quantizer
        assert (
            self.quantizer.embedding_dim == embedding_dim
            if self.quantizer is not None
            else True
        )

        # Model outputs ----------------------------------------------------------------

        self.output_activation = output_activation or nn.Identity()
        self.output_activation_extension = output_activation_extension

        # Loss calculation ------------------------------------------------------------

        self.reconstruction_loss = reconstruction_loss
        self.quantization_loss_weight = quantisation_loss_weight

        # Loss optimisation ------------------------------------------------------------

        self.lr = lr
        self.lr_warmup = lr_warmup
        self.begin_cosine_annealing = begin_cosine_annealing
        self.lr_cosine_period = lr_cosine_period

        # Model weights EMA ------------------------------------------------------------

        self.ema_model_weights = ema_model_weights
        self.ema_decay = ema_decay
        self.ema_update_freq = ema_update_freq

        # ------------------------------------------------------------------------------

    @property
    def modality(self) -> Type[LegacySurveySegmentationMap]:
        return LegacySurveySegmentationMap

    @property
    def quantizer(self) -> Optional[Quantizer]:
        return self._quantizer

    def _encode(self, x: LegacySurveySegmentationMap) -> Float[Tensor, "b c h*w"]:
        # Extract the field tensor from the ScalarField modality
        field_tensor = x.field

        # Add channel dimension if needed (ScalarField is batch x height x width)
        if field_tensor.dim() == 3:
            field_tensor = field_tensor.unsqueeze(1)  # Add channel dimension

        # Apply center cropping to 96x96
        processed_field = self.center_crop(field_tensor)

        h = self.encoder(processed_field)
        h = self.encode_proj(h)
        h = h.reshape(h.shape[0], h.shape[1], -1)
        return h

    def _decode(self, z: Float[Tensor, "b c h*w"]) -> LegacySurveySegmentationMap:
        batch_size, embedding_dim, n_tokens = z.shape
        spatial_size = int(n_tokens**0.5)
        assert spatial_size * spatial_size == n_tokens, (
            f"n_tokens ({n_tokens}) is not a perfect square. "
            f"Calculated spatial_size: {spatial_size}."
        )
        z = z.reshape(batch_size, embedding_dim, spatial_size, spatial_size)
        h = self.decode_proj(z)
        x_hat = self.decoder(h)
        x_hat = self._output_activation(x_hat).clip(0.0, 1.0)

        # Remove channel dimension for ScalarField (expects batch x height x width)
        if x_hat.shape[1] == 1:
            x_hat = x_hat.squeeze(1)  # Remove channel dimension

        return LegacySurveySegmentationMap(field=x_hat)

    def _output_activation(
        self,
        x_hat: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, "b c h w"]:
        x_hat = self.output_activation(x_hat)

        d = self.output_activation_extension
        if d is not None:
            x_hat = (1 + 2 * d) * x_hat - d

        return x_hat


# ======================================================================================
# Specific subclasses of AutoencoderScalarFieldCodec
# ======================================================================================


class ScalarFieldCodec(AutoencoderScalarFieldCodec, PyTorchModelHubMixin):
    """Convolutional autoencoder codec for scalar fields."""

    def __init__(
        # ------------------------------------------------------------------------------
        self,
        # Code dimensions --------------------------------------------------------------
        encoder_output_dim: int,
        decoder_input_dim: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        # Encoder / decoder architecture -----------------------------------------------
        res_hidden_dims: int = 64,
        num_res_layers: int = 2,
        num_downsamples: int = 3,
        # VAE operation ----------------------------------------------------------------
        variational: bool = False,
        # Model outputs ----------------------------------------------------------------
        output_activation: Optional[Callable] = F.sigmoid,
        output_activation_extension: Optional[float] = None,
        # Loss calculation -------------------------------------------------------------
        reconstruction_loss: Callable = F.mse_loss,
        quantisation_loss_weight: float = 1.0,
        # Loss optimisation ------------------------------------------------------------
        lr: float = 1e-3,
        lr_warmup: Optional[int] = None,
        begin_cosine_annealing: Optional[int] = None,
        lr_cosine_period: Optional[int] = None,
        # Model weights EMA ------------------------------------------------------------
        ema_model_weights: bool = False,
        ema_decay: float = 0.9999,
        ema_update_freq: int = 1,
        levels=[8, 5, 5, 5],
        # ------------------------------------------------------------------------------
    ):
        super().__init__(
            encoder_output_dim=encoder_output_dim,
            decoder_input_dim=decoder_input_dim,
            embedding_dim=embedding_dim,
            variational=variational,
            output_activation=output_activation,
            output_activation_extension=output_activation_extension,
            reconstruction_loss=reconstruction_loss,
            quantisation_loss_weight=quantisation_loss_weight,
            lr=lr,
            lr_warmup=lr_warmup,
            begin_cosine_annealing=begin_cosine_annealing,
            lr_cosine_period=lr_cosine_period,
            ema_model_weights=ema_model_weights,
            ema_decay=ema_decay,
            ema_update_freq=ema_update_freq,
        )

        self._quantizer = FiniteScalarQuantizer(levels=levels)

        # Encoder ----------------------------------------------------------------------
        self.encoder = Encoder2d(
            in_dims=1,
            out_dims=self.encoder_output_dim,
            res_hidden_dims=res_hidden_dims,
            num_res_layers=num_res_layers,
            num_downsamples=num_downsamples,
        )

        # Decoder ----------------------------------------------------------------------
        self.decoder = Decoder2d(
            in_dims=self.decoder_input_dim,  # = encoder_output_dim unless overridden
            out_dims=1,
            hidden_dims=self.decoder_input_dim,
            res_hidden_dims=res_hidden_dims,
            num_res_layers=num_res_layers,
            num_upsamples=num_downsamples,
        )

        # Model weights EMA ------------------------------------------------------------
        if ema_model_weights:
            self.ema = ModelEmaV2(self, decay=ema_decay, device=None)
        else:
            self.ema = None
