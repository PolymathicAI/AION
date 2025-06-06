import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float


class SubsampledLinear(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, subsample_in: bool = True):
        """
        Subsampled linear layer for the encoder.
        It takes in a zero-padded tensor and a mask.
        It projects the tensor into some shared projection space.
        It can also be used to reverse out of the space with the mask.

        Args:
            dim_in : Number of total possible bands.
            dim_out : Number of embedding dimensions.
            subsample_in : Whether to subsample the input. Defaults to True.
        """
        super().__init__()
        self.subsample_in = subsample_in
        self.dim_in = dim_in  # Number of total possible bands
        self.dim_out = dim_out  # Number of embedding dimensions
        temp_linear = torch.nn.Linear(dim_in, dim_out)
        self.weight = torch.nn.Parameter(temp_linear.weight)
        self.bias = torch.nn.Parameter(temp_linear.bias)

    def _subsample_in(self, x, labels: Bool[torch.Tensor, " b c"]):
        # Get mask
        mask = labels[:, None, None, :].float()
        x = x * mask

        # Normalize
        label_sizes = labels.sum(dim=1, keepdim=True)
        scales = ((self.dim_in / label_sizes) ** 0.5).squeeze()

        # Apply linear layer
        return scales[:, None, None, None] * F.linear(x, self.weight, self.bias)

    def _subsample_out(self, x, labels):
        # Get mask
        mask = labels[:, None, None, :].float()

        # Apply linear layer and mask
        return F.linear(x, self.weight, self.bias) * mask

    def forward(
        self, x: Float[torch.Tensor, " b c h w"], labels: Bool[torch.Tensor, " b c"]
    ) -> Float[torch.Tensor, " b c h w"]:
        x = rearrange(x, "b c h w -> b h w c")

        if self.subsample_in:
            x = self._subsample_in(x, labels)

        else:
            x = self._subsample_out(x, labels)

        x = rearrange(x, "b h w c -> b c h w")

        return x
