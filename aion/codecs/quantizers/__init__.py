import math
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float, Integer
from .lookup_free_quantization import LFQ


class Quantizer(torch.nn.Module, ABC):
    """Abstract interface for all quantizer modules."""

    @abstractmethod
    def quantize(
        self, x: Float[torch.Tensor, " b c1 *input_shape"]
    ) -> Float[torch.Tensor, " b c *code_shape"]:
        """Quantize the input tensor."""
        raise NotImplementedError

    @abstractmethod
    def decode(
        self, z: Float[torch.Tensor, " b c *code_shape"]
    ) -> Float[torch.Tensor, " b c *input_shape"]:
        """Reconstruct the input tensor from the quantized tensor."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, z_e: Float[torch.Tensor, " b c *input_shape"]
    ) -> tuple[
        Float[torch.Tensor, " b c *code_shape"],
        Float[torch.Tensor, " b"],
        Float[torch.Tensor, " b"],
    ]:
        """Performs a forward pass through the vector quantizer.
        Args:
            x: The input tensor to be quantized.
        Returns:
            z: The quantized tensor.
            quantization_error: The error of the quantization.
            codebook_usage: The fraction of codes used in the codebook.
        """
        raise NotImplementedError


class FiniteScalarQuantizer(Quantizer):
    def __init__(
        self,
        levels: list[int],
        eps: float = 1e-3,
    ):
        """Finite scalar quantizer (FSQ) module
        https://arxiv.org/pdf/2309.15505.pdf

        Following the implementation from:
        https://github.com/duchenzhuang/FSQ-pytorch/blob/main/quantizers/fsq.py

        Args:
            levels: list[int]
                The number of levels for each dimension. Length of the list should match
                the number of embedding dimensions.
            eps: float
                The epsilon value for the FSQ.
        """
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("levels", _levels)
        self._embedding_dim = len(levels)
        self._basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.eps = eps

    @property
    def codebook_size(self):
        return math.prod(self.levels)

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def _bound(
        self, z: Float[torch.Tensor, " b t *c"]
    ) -> Float[torch.Tensor, " b t *c"]:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self.levels - 1) * (1 + self.eps) / 2
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def _quantize(
        self, z: Float[torch.Tensor, " b t *c"]
    ) -> Float[torch.Tensor, " b t *c"]:
        """Quantizes z, returns quantized codes zhat with the same shape as z.
        Assumes last dimension of z is the embedding dimension.
        """

        def round_ste(z):
            zhat = z.round()
            return z + (zhat - z).detach()

        quantized = round_ste(self._bound(z))
        # Renormalize to [-1, 1].
        half_width = self.levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self.levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self.levels // 2
        return (zhat - half_width) / half_width

    def quantize(
        self, z: Float[torch.Tensor, " b *c t"]
    ) -> Float[torch.Tensor, " b *c t"]:
        """
        Quantizes the input tensor.

        Args:
            z (Tensor): The input tensor to be quantized.

        Returns:
            Tensor: The quantized tensor, same shape as input.
        """
        # Move the embedding dimension to the last dimension for easier broadcasting
        z = z.moveaxis(1, -1)
        zhat = self._quantize(z)
        return zhat.moveaxis(-1, 1)

    def encode(
        self, z: Float[torch.Tensor, " b *c t"]
    ) -> Integer[torch.Tensor, " b *code"]:
        """
        Encodes the input tensor `z` using quantization.

        Args:
            z (Tensor): The input tensor to be encoded.

        Returns:
            Tensor: integer code index.
        """
        # Move the embedding dimension to the last dimension for easier broadcasting
        z = z.moveaxis(1, -1)
        zhat = self._quantize(z)
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis.to(zhat)).sum(axis=-1).to(torch.int32)

    def decode(
        self, codes: Integer[torch.Tensor, " b *code"]
    ) -> Float[torch.Tensor, "b *c t"]:
        """
        Decodes the given codes into the corresponding values.

        Args:
            codes (Tensor): The codes to be decoded.

        Returns:
            Tensor: The decoded tensor.
        """
        indices = codes.unsqueeze(-1)
        codes_non_centered = (indices // self._basis.to(indices)) % self.levels
        zhat = self._scale_and_shift_inverse(codes_non_centered)
        # Move the embedding dimension back to the second dimension
        return zhat.moveaxis(-1, 1)

    def forward(
        self, z_e: Float[torch.Tensor, " b t *codes"]
    ) -> tuple[
        Float[torch.Tensor, " b t *shape"],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        """
        Forward pass of the quantizer module.

        Args:
            z_e: The input tensor.

        Returns:
            tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - decoded (Tensor): The decoded tensor.
                - loss (Tensor): In this case, no additional loss is necessary, so always returns 0.
                - codebook_usage (Tensor): The ratio of unique codes used in the codebook.
        """
        z_q = self.quantize(z_e)
        codes = self.encode(z_e)
        codebook_usage = len(torch.unique(codes)) / self.codebook_size
        return z_q, torch.zeros([]), codebook_usage


class LucidrainsLFQ(Quantizer):
    def __init__(
        self,
        dim: int | None = None,
        codebook_size: int | None = None,
        inv_temperature: float = 100.0,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
        diversity_gamma: float = 1.0,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: bool | None = None,
        codebook_scale: float = 1.0,
        frac_per_sample_entropy: float = 1.0,
        use_code_agnostic_commit_loss: bool = False,
        projection_has_bias: bool = True,
        soft_clamp_input_value: bool | None = None,
        cosine_sim_project_in: bool = False,
        cosine_sim_project_in_scale: float | None = None,
    ):
        """Lookup Free Quantizer (LFQ) from the MagVITv2 paper
        https://arxiv.org/abs/2310.05737

        Following the implementation from vector-quantize-pytorch
        """
        super().__init__()
        self._inverse_temperature = inv_temperature
        self._quantizer = LFQ(
            dim=dim,
            codebook_size=codebook_size,
            entropy_loss_weight=entropy_loss_weight,
            commitment_loss_weight=commitment_loss_weight,
            diversity_gamma=diversity_gamma,
            num_codebooks=num_codebooks,
            keep_num_codebooks_dim=keep_num_codebooks_dim,
            codebook_scale=codebook_scale,
            frac_per_sample_entropy=frac_per_sample_entropy,
            use_code_agnostic_commit_loss=use_code_agnostic_commit_loss,
            projection_has_bias=projection_has_bias,
            soft_clamp_input_value=soft_clamp_input_value,
            cosine_sim_project_in=cosine_sim_project_in,
            cosine_sim_project_in_scale=cosine_sim_project_in_scale,
        )

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the vector quantizer.
        Args:
            z_e: Tensor (B, C, ...)
                The input tensor to be quantized.
        Returns:
            z_q: Tensor
                The quantized tensor.
            loss: Tensor
                The embedding loss for the quantization.
            codebook_usage: Tensor
                The fraction of codes used in the codebook.
        """
        # In cases where we only have a sequence, we need to move the sequence dimension to the last dimension
        # For compatibility with the upstream quantizer
        if len(z_e.shape) == 3:
            z_e = z_e.movedim(1, -1)
        z_q, indices, aux_loss = self._quantizer(
            z_e, inv_temperature=self._inverse_temperature
        )
        codebook_usage = indices.unique().numel() / self.codebook_size
        if len(z_q.shape) == 3:
            z_q = z_q.movedim(-1, 1)
        return z_q, aux_loss, torch.tensor(codebook_usage)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes the input tensor z, returns the corresponding
        codebook index.
        """
        return self.encode(z)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor z, returns the corresponding
        codebook index.
        """
        # In cases where we only have a sequence, we need to move the sequence dimension to the last dimension
        # For compatibility with the upstream quantizer
        if len(z.shape) == 3:
            z = z.movedim(1, -1)
        z_q, indices, aux_loss = self._quantizer(
            z, inv_temperature=self._inverse_temperature
        )
        return indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decodes the input code index into corresponding codebook entry of
        dimension (embedding_dim).
        """
        z = self._quantizer.indices_to_codes(codes)
        # For compatibility with the upstream quantizer, we need to move the last dimension to the sequence dimension
        if len(z.shape) == 3:
            z = z.movedim(-1, 1)
        return z

    @property
    def codebook_size(self) -> int:
        """Returns the size of the codebook."""
        return len(self._quantizer.codebook)

    @property
    def embedding_dim(self) -> int:
        """Returns the dimension of the codebook entries."""
        return self._quantizer.codebook_dim


class ScalarLinearQuantizer(Quantizer):
    """A simple non-adaptive quantizer which will encode scalars by binning
    on fixed histogram in the specified range.
    """

    def __init__(self, codebook_size: int, range: tuple[float, float]):
        super().__init__()
        self.register_buffer(
            "buckets", torch.linspace(range[0], range[1], codebook_size)
        )

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the vector quantizer.
        Args:
            z_e: Tensor (B, C, ...)
                The input tensor to be quantized.
        Returns:
            z_q: Tensor
                The quantized tensor.
            loss: Tensor
                The embedding loss for the quantization.
            codebook_usage: Tensor
                The fraction of codes used in the codebook.
        """
        indices = self.encode(z_e)
        z_q = self.decode(indices)
        codebook_usage = indices.unique().numel() / self.codebook_size
        return z_q, torch.tensor(0), torch.tensor(codebook_usage)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes the input tensor z, returns the corresponding
        codebook index.
        """
        return self.decode(self.encode(z))

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor z, returns the corresponding
        codebook index.
        """
        return torch.clamp(
            torch.bucketize(z, self.buckets, out_int32=True), 0, self.codebook_size - 1
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decodes the input code index into corresponding codebook entry of
        dimension (embedding_dim).
        """
        return self.buckets[codes]

    @property
    def codebook_size(self) -> int:
        """Returns the size of the codebook."""
        return len(self.buckets)

    @property
    def codebook(self) -> torch.Tensor:
        """Returns the codebook."""
        return self.decode(torch.arange(self.codebook_size))

    @property
    def embedding_dim(self) -> int:
        """Returns the dimension of the codebook entries."""
        return 1
