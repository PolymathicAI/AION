import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class SameConv2d(torch.nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 2)
        padding = [k // 2 for k in kernel_size]
        self.conv = torch.nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class SqueezeExcite(torch.nn.Module):
    # global context network - attention-esque squeeze-excite variant (https://arxiv.org/abs/2012.13375)

    def __init__(self, dim, *, dim_out=None, dim_hidden_min=16, init_bias=-10):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim

        self.to_k = torch.nn.Conv2d(dim, 1, 1)
        dim_hidden = max(dim_hidden_min, dim_out // 2)

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_hidden, 1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(dim_hidden, dim_out, 1),
            torch.nn.Sigmoid(),
        )

        torch.nn.init.zeros_(self.net[-2].weight)
        torch.nn.init.constant_(self.net[-2].bias, init_bias)

    def forward(self, x):
        context = self.to_k(x)

        context = rearrange(context, "b c h w -> b c (h w)").softmax(dim=-1)
        spatial_flattened_input = rearrange(x, "b c h w -> b c (h w)")

        out = torch.einsum("b i n, b c n -> b c i", context, spatial_flattened_input)
        out = rearrange(out, "... -> ... 1")
        gates = self.net(out)

        return gates * x


class ResidualUnit(torch.nn.Module):
    def __init__(self, dim: int, kernel_size: int | tuple[int, int, int]):
        super().__init__()
        self.net = torch.nn.Sequential(
            SameConv2d(dim, dim, kernel_size),
            torch.nn.ELU(),
            torch.nn.Conv2d(dim, dim, 1),
            torch.nn.ELU(),
            SqueezeExcite(dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x) + x


class SpatialDownsample2x(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.conv = torch.nn.Conv2d(
            dim, dim_out, kernel_size, stride=2, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out


class SpatialUpsample2x(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int = None):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        conv = torch.nn.Conv2d(dim, dim_out * 4, 1)

        self.net = torch.nn.Sequential(
            conv,
            torch.nn.SiLU(),
            Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2),
        )

        self.init_conv_(conv)

    def init_conv_(self, conv: torch.nn.Module):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        torch.nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        conv.weight.data.copy_(conv_weight)
        torch.nn.init.zeros_(conv.bias.data)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class MagVitAE(torch.nn.Module):
    """MagViTAE implementation from Yu, et al. (2024), adapted for Pytorch.
    Code borrowed from https://github.com/lucidrains/magvit2-pytorch, and adapted for images.
    """

    def __init__(
        self,
        n_bands: int = 3,
        hidden_dims: int = 512,
        residual_conv_kernel_size: int = 3,
        n_compressions: int = 2,
        num_consecutive: int = 2,
    ):
        super().__init__()

        self.encoder_layers = torch.nn.ModuleList([])
        self.decoder_layers = torch.nn.ModuleList([])
        init_dim = int(hidden_dims / 2**n_compressions)
        dim = init_dim

        self.conv_in = SameConv2d(n_bands, init_dim, 7)
        self.conv_out = SameConv2d(init_dim, n_bands, 3)

        # Residual layers
        encoder_layer = ResidualUnit(dim, residual_conv_kernel_size)
        decoder_layer = ResidualUnit(dim, residual_conv_kernel_size)
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        # Compressions
        for i in range(n_compressions):
            dim_out = dim * 2
            encoder_layer = SpatialDownsample2x(dim, dim_out)
            decoder_layer = SpatialUpsample2x(dim_out, dim)
            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)
            dim = dim_out

            # Consecutive residual layers
            encoder_layer = torch.nn.Sequential(
                *[
                    ResidualUnit(dim, residual_conv_kernel_size)
                    for _ in range(num_consecutive)
                ]
            )
            decoder_layer = torch.nn.Sequential(
                *[
                    ResidualUnit(dim, residual_conv_kernel_size)
                    for _ in range(num_consecutive)
                ]
            )
            self.encoder_layers.append(encoder_layer)
            self.decoder_layers.insert(0, decoder_layer)

        # Add a final non-compress layer
        dim_out = dim
        encoder_layer = SameConv2d(dim, dim_out, 7)
        decoder_layer = SameConv2d(dim_out, dim, 3)
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)
        dim = dim_out

        # Consecutive residual layers
        encoder_layer = torch.nn.Sequential(
            *[
                ResidualUnit(dim, residual_conv_kernel_size)
                for _ in range(num_consecutive)
            ]
        )
        decoder_layer = torch.nn.Sequential(
            *[
                ResidualUnit(dim, residual_conv_kernel_size)
                for _ in range(num_consecutive)
            ]
        )
        self.encoder_layers.append(encoder_layer)
        self.decoder_layers.insert(0, decoder_layer)

        # add a final norm just before quantization layer
        self.encoder_layers.append(
            torch.nn.Sequential(
                Rearrange("b c ... -> b ... c"),
                torch.nn.LayerNorm(dim),
                Rearrange("b ... c -> b c ..."),
            )
        )

    def encode(self, x: torch.Tensor):
        x = self.conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor):
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.conv_out(x)
        return x
