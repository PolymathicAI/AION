import math

import torch
from jaxtyping import Float, Integer

from aion.codecs.quantizers.base import Quantizer


class IdentityQuantizer(Quantizer):
    """
    Identity quantizer module.

    The identity quantizer module takes a batch of tensors and returns the same tensor.

    Args:
        codebook_size: int
            The number of labels to be used as signature for the codebook.
    """

    def __init__(self, codebook_size: int):
        super().__init__()
        self._codebook_size = codebook_size

    def forward(
        self, z_e: Float[torch.Tensor, " b c"]
    ) -> tuple[
        Float[torch.Tensor, " b c"],
        Float[torch.Tensor, " b"],
        Float[torch.Tensor, " b"],
    ]:
        """Performs a forward pass through the vector quantizer.
        Args:
            z_e: The input tensor to be quantized.
        Returns:
            z_q: The quantized tensor.
            loss: The embedding loss for the quantization.
            codebook_usage: The fraction of codes used in the codebook.
        """
        codebook_usage = z_e.unique().numel() / self._codebook_size.item()
        return self.quantize(z_e), torch.tensor(0), torch.tensor(codebook_usage)

    def quantize(self, z: Float[torch.Tensor, " b c"]) -> Float[torch.Tensor, " b c"]:
        """Quantize the input tensor z, returns corresponding
        codebook entry.
        """
        return z

    def encode(self, z: Float[torch.Tensor, " b c"]) -> Integer[torch.Tensor, " b c"]:
        return self.quantize(z)

    def reconstruct(
        self, codes: Float[torch.Tensor, " b c"]
    ) -> Float[torch.Tensor, " b c"]:
        """Decodes the input code index into corresponding codebook entry of
        dimension (embedding_dim).
        """
        return codes

    @property
    def codebook_size(self) -> int:
        """Returns the size of the codebook."""
        return int(self._codebook_size.item())

    @property
    def codebook(self) -> Float[torch.Tensor, " c"]:
        """Returns the codebook."""
        return torch.arange(self._codebook_size.item())

    @property
    def embedding_dim(self) -> int:
        """Returns the dimension of the codebook entries."""
        return 1


class FiniteScaleQuantizer(Quantizer):
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

    def reconstruct(
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
