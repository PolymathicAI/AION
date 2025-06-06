import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable
from einops import rearrange
from jaxtyping import Float


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.t_emb_proj = zero_init(nn.Linear(embedding_dim, input_dim * 2))

    def forward(
        self,
        x: Float[torch.Tensor, "b ... c"],
        t_emb: Float[torch.Tensor, "b d"],
    ) -> Float[torch.Tensor, "b ... c"]:
        c = x.shape[-1]
        num_spatial_dims = len(x.shape) - 2
        assert c == self.input_dim, "input_dim must match the last dimension of x"

        t_emb = self.t_emb_proj(t_emb)
        new_shape = [t_emb.shape[0]] + ([1] * num_spatial_dims) + [-1]
        t_emb = t_emb.reshape(*new_shape)
        scale, shift = t_emb.chunk(2, dim=-1)

        x = F.layer_norm(x, [c])

        x = x * (1 + scale) + shift

        return x


CONV_FUNCTION: Dict[str, Callable] = {
    "1d": nn.Conv1d,
    "1dT": nn.ConvTranspose2d,
    "1dB": nn.BatchNorm1d,
    "2d": nn.Conv2d,
    "2dT": nn.ConvTranspose2d,
    "2dB": nn.BatchNorm2d,
}


class ResLayer(nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims, conv_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            CONV_FUNCTION[conv_dim](
                in_dims, hidden_dims, kernel_size=3, stride=1, padding=1
            ),
            # TODO
            CONV_FUNCTION[f"{conv_dim}B"](hidden_dims),
            nn.ReLU(),
            CONV_FUNCTION[conv_dim](
                hidden_dims, out_dims, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(self, x):
        x = x + self.net(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims, num_res_layers, conv_dim):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims

        self.num_res_layers = num_res_layers
        self.net = nn.Sequential(
            *[
                ResLayer(in_dims, out_dims, hidden_dims, conv_dim=conv_dim)
                for _ in range(num_res_layers)
            ],
        )

    def forward(self, x):
        return self.net(x)


def zero_init(layer):
    """Initializes the weights of the layer to zero."""
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, maxpool: bool = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2) if maxpool else Downsample2d(in_channels, in_channels),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(
        self, x: Float[torch.Tensor, "b c h w"]
    ) -> Float[torch.Tensor, "b c h w"]:
        return self.conv(x)


class Upsample2d(nn.Module):
    """A 2D upsampling layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv_transpose: bool = False,
        kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_transpose = use_conv_transpose

        self.norm = nn.LayerNorm(in_channels)

        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            self.conv = nn.ConvTranspose2d(
                self.in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
            )
        else:
            if kernel_size is None:
                kernel_size = 3
            self.conv = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=1,
            )

    def forward(
        self,
        x: Float[torch.Tensor, "b c h w"],
    ) -> Float[torch.Tensor, "b c h w"]:
        b, c, h, w = x.shape

        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        if self.use_conv_transpose:
            return self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear")
            x = self.conv(x)

        return x


class Downsample2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.norm = nn.LayerNorm(in_channels)

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "b c h w"],
    ) -> Float[torch.Tensor, "b c h w"]:
        b, c, h, w = x.shape

        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        x = self.conv(x)
        return x


class ResNetBlock2d(nn.Module):
    r"""
    A Resnet block with optionally conditional normalization.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        conditional_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        non_linearity (`str`, *optional*, default to `"gelu"`): the activation function to use.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditional_channels: int = 0,
        dropout: float = 0.0,
        shortcut: bool = True,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.conditional_channels = conditional_channels

        if self.conditional_channels == 0:
            self.register_buffer("fake_t_emb", torch.zeros(1, 1))

        self.norm1 = AdaLayerNorm(in_channels, max(conditional_channels, 1))

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.norm2 = AdaLayerNorm(out_channels, max(conditional_channels, 1))

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = zero_init(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.upsample = self.downsample = None

        if self.up:
            self.upsample = Upsample2d(in_channels)
        elif self.down:
            self.downsample = Downsample2d(in_channels)

        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if shortcut
            else nn.Identity()
        )

    def forward(
        self,
        x: Float[torch.Tensor, "b c h w"],
        t_emb: Optional[Float[torch.Tensor, "b n"]] = None,
    ) -> Float[torch.Tensor, "b c h w"]:
        out = x

        if t_emb is None:
            assert self.conditional_channels == 0
            t_emb = self.fake_t_emb.to(x)

        out = self.norm1(out.permute(0, 2, 3, 1), t_emb).permute(
            0, 3, 1, 2
        )  # b h w c -> b h w c -> b c h w

        out = F.gelu(out)

        if self.upsample is not None:
            x = self.upsample(x)
            out = self.upsample(out)

        elif self.downsample is not None:
            x = self.downsample(x)
            out = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out.permute(0, 2, 3, 1), t_emb).permute(0, 3, 1, 2)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return self.conv_shortcut(x) + out


class ResEncoder2D(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.encoder = nn.Sequential(
            ResNetBlock2d(input_dims, output_dims // 4, down=True),
            ResNetBlock2d(output_dims // 4, output_dims // 2, down=True),
            ResNetBlock2d(output_dims // 2, output_dims, down=True),
        )

    def forward(self, x):
        return self.encoder(x)


class ResDecoder2D(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.decoder = nn.Sequential(
            ResNetBlock2d(input_dims, input_dims // 2, up=True),
            ResNetBlock2d(input_dims // 2, input_dims // 4, up=True),
            ResNetBlock2d(input_dims // 4, output_dims, up=True),
        )

    def forward(self, x):
        return self.decoder(x)


class Encoder2d(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        res_hidden_dims: int,
        num_res_layers: int,
        num_downsamples: int = 3,
    ):
        """
        VQ-VAE encoder, transcribed from Deepmind's Sonnet implementation (https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb)

        Args:
        in_dims: int
            The number of input channels.
        out_dims: int
            The number of output channels.
        num_downsamples: int
            The number of downsamples to apply to the input.
        res_hidden_dims: int
            The number of hidden channels in the residual block.
        num_res_layers: int
            The number of residual layers in the residual block.
        """
        super().__init__()

        _dims = [
            in_dims,
            *(out_dims // 2**i for i in range(num_downsamples - 1, 0, -1)),
            out_dims,
        ]
        downsampling_conv_layers = []
        for i in range(num_downsamples):
            downsampling_conv_layers.append(
                nn.Conv2d(
                    in_channels=_dims[i],
                    out_channels=_dims[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            downsampling_conv_layers.append(nn.ReLU())

        self.net = nn.Sequential(
            *downsampling_conv_layers,
            nn.Conv2d(
                in_channels=out_dims,
                out_channels=out_dims,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResBlock(
                in_dims=out_dims,
                out_dims=out_dims,
                hidden_dims=res_hidden_dims,
                num_res_layers=num_res_layers,
                conv_dim="2d",
            ),
        )

    def forward(
        self, x: Float[torch.Tensor, "b c h w"]
    ) -> Float[torch.Tensor, "b c h w"]:
        return self.net(x)


class Decoder2d(nn.Module):
    """
    VQ-VAE decoder, transcribed from Deepmind's Sonnet implementation (https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb)
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        hidden_dims: int,
        res_hidden_dims: int,
        num_res_layers: int,
        num_upsamples: int = 3,
    ):
        """
        Initialize the decoder.

        Inputs:
        =======
        in_dims: int
            The number of input channels (i.e. the number of channels in the latent space).
        out_dims: int
            The number of output channels.
        hidden_dims: int
            The number of hidden channels in the decoder.
        res_hidden_dims: int
            The number of hidden channels in the residual block.
        num_res_layers: int
            The number of residual layers in the residual block.
        """
        super().__init__()

        _dims = [
            hidden_dims,
            *(hidden_dims // 2**i for i in range(num_upsamples - 1, 0, -1)),
            out_dims,
        ]

        upsampling_conv_layers = []
        for i in range(num_upsamples):
            upsampling_conv_layers.append(nn.ReLU())
            upsampling_conv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=_dims[i],
                    out_channels=_dims[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_dims,
                out_channels=hidden_dims,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResBlock(
                in_dims=hidden_dims,
                out_dims=hidden_dims,
                hidden_dims=res_hidden_dims,
                num_res_layers=num_res_layers,
                conv_dim="2d",
            ),
            *upsampling_conv_layers,
        )

    def forward(
        self, x: Float[torch.Tensor, "b c h w"]
    ) -> Float[torch.Tensor, "b c h w"]:
        return self.net(x)
