import torch

from aion.codecs.modules.utils import LayerNorm, GRN


class ConvNextBlock1d(torch.nn.Module):
    """ConvNeXtV2 Block.
    Modified to 1D from the original 2D implementation from https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = torch.nn.Linear(4 * dim, dim)

    def forward(self, x):
        y = self.dwconv(x)
        y = y.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.grn(y)
        y = self.pwconv2(y)
        y = y.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)

        y = x + y
        return y


class ConvNextEncoder1d(torch.nn.Module):
    r"""ConvNeXt encoder.

    Modified from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Args:
        in_chans : Number of input image channels. Default: 3
        depths : Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims : Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate : Stochastic depth rate. Default: 0.
        layer_scale_init_value : Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        in_chans: int = 2,
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (96, 192, 384, 768),
    ):
        super().__init__()
        assert len(depths) == len(dims), "depths and dims should have the same length"
        num_layers = len(depths)

        self.downsample_layers = (
            torch.nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(num_layers - 1):
            downsample_layer = torch.nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                torch.nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = torch.nn.ModuleList()
        for i in range(num_layers):
            stage = torch.nn.Sequential(
                *[
                    ConvNextBlock1d(
                        dim=dims[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for ds, st in zip(self.downsample_layers, self.stages):
            x = ds(x)
            x = st(x)
        return self.norm(x)


class ConvNextDecoder1d(torch.nn.Module):
    r"""ConvNeXt decoder. Essentially a mirrored version of the encoder.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        in_chans=768,
        depths=[3, 3, 9, 3],
        dims=[384, 192, 96, 2],
    ):
        super().__init__()
        assert len(depths) == len(dims), "depths and dims should have the same length"
        num_layers = len(depths)

        self.upsample_layers = torch.nn.ModuleList()

        stem = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.upsample_layers.append(stem)

        for i in range(num_layers - 1):
            upsample_layer = torch.nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                torch.nn.ConvTranspose1d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2 if i < (num_layers - 2) else 4,
                    stride=2 if i < (num_layers - 2) else 4,
                ),
            )
            self.upsample_layers.append(upsample_layer)

        self.stages = torch.nn.ModuleList()
        for i in range(num_layers):
            stage = torch.nn.Sequential(
                *[
                    ConvNextBlock1d(
                        dim=dims[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for us, st in zip(self.upsample_layers, self.stages):
            x = us(x)
            x = st(x)
        return x
