from abc import ABC, abstractmethod

import torch
from jaxtyping import Float


class Quantizer(torch.nn.Module, ABC):
    """Abstract interface for all quantizer modules."""

    @abstractmethod
    def quantize(
        self, x: Float[torch.Tensor, " b c1 *input_shape"]
    ) -> Float[torch.Tensor, " b c *code_shape"]:
        """Quantize the input tensor."""
        raise NotImplementedError

    @abstractmethod
    def reconstruct(
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
