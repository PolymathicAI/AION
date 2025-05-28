import torch
from pathlib import Path

from aion.model import AION
from aion.fourm.fm_utils import NormCrossAttention


class BaseModel(torch.nn.Module):
    def __init__(self, aion_model_path: str | Path, num_encoder_tokens: int = 576):
        super().__init__()
        self.aion_backbone = AION.from_pretrained(aion_model_path)
        self.aion_backbone.freeze_encoder()
        self.aion_backbone.freeze_decoder()
        self.aion_backbone = torch.compile(self.aion_backbone)
        self.num_encoder_tokens = num_encoder_tokens

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute AION embeddings of the input `x`.
        `x` is typically the output of the `CodecManager.encode` method.
        """
        with torch.no_grad():
            embeddings = self.aion_backbone.encode(
                x, num_encoder_tokens=self.num_encoder_tokens
            )
        return embeddings


class LinearProbing(BaseModel):
    def __init__(
        self, dim_out: int, aion_model_path: str | Path, num_encoder_tokens: int = 576
    ):
        super().__init__(aion_model_path, num_encoder_tokens)
        self.linear = torch.nn.Linear(self.aion_backbone.dim, dim_out)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = super().forward(x)
        embeddings = torch.mean(embeddings, dim=1)
        output = self.linear(embeddings)
        return output


class CrossAttentionProbing(BaseModel):
    def __init__(
        self,
        dim_out: int,
        num_heads: int,
        aion_model_path: str | Path,
        num_encoder_tokens: int = 576,
    ):
        super().__init__(aion_model_path, num_encoder_tokens)
        self.query = torch.nn.Parameter(torch.randn(1, dim_out, self.aion_backbone.dim))
        self.attention = NormCrossAttention(
            self.aion_backbone.dim, num_heads=num_heads, proj_bias=False
        )
        self.attention = torch.compile(self.attention)
        self.decoders = torch.nn.ModuleList(
            [torch.nn.Linear(self.aion_backbone.dim, 1) for _ in range(dim_out)]
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = super().forward(x)
        # Apply cross-attention
        query = self.query.expand(embeddings.size(0), -1, -1)
        output = self.attention(query, embeddings)
        # Apply linear layers
        # TODO: Optimize operation instead of for-loop
        output = torch.cat(
            [decoder(output[:, i]) for i, decoder in enumerate(self.decoders)], dim=-1
        )
        return output
