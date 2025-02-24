# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
from einops import repeat

from .flow import NSF, BetaSchedule, MLPBackbone, RectifiedFlow
from .fm_utils import build_1d_sincos_posemb, build_2d_sincos_posemb, pair


class SequenceDecoderEmbedding(nn.Module):
    """Embedding module for sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        share_embedding: Set to True to share input and output embedding weights
    """
    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 share_embedding: bool = True,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.share_embedding = share_embedding

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes

        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}")
            # Get all posembs, than truncate up to max length
            pos_emb = build_1d_sincos_posemb(max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens)[:self.max_length]
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_length, self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens, padding_idx=self.padding_idx)

        # Output projection layer
        self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

        if self.share_embedding:
            # Share input and output embedding weights
            self.to_logits.weight = self.token_emb.weight


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict, with at least the following keys:
                - 'tensor' (torch.Tensor): Token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'target_mask' (torch.Tensor): Mask for valid tokens in the target sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the target sequence. Shape (B, L, D).
                - 'ids' (torch.Tensor): Original token sequence from input dict. Shape (B, L).
        """
        ids = d['tensor']
        B = ids.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'

        # Map to embedding
        x = self.token_emb(ids)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)

        # Target pos encoding
        target_mask = d['target_mask']
        target_pos_id = (~target_mask).int().cumsum(dim=1) - 1
        target_pos_id[target_mask] = 0
        # Sometimes target sequence is over max length, it will be truncated in decoder
        target_pos_id[target_pos_id >= self.max_length] = 0
        target_pos_emb = torch.gather(expanded_pos_emb, dim=1, index=repeat(target_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]))
        target_pos_emb[target_mask] = 0

        x_emb = target_pos_emb + self.mod_emb


        d['x'] = x
        d['emb'] = x_emb
        d['ids'] = d['tensor']

        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)
        
        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, M, V)
        """
        logits = self.to_logits(x)
        return logits



class ImageTokenDecoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
        share_embedding: Set to True to share input and output embedding weights
    """
    def __init__(self,
                 vocab_size: int,
                 patch_size: Union[int, Tuple[int,int]] = 16,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 image_size: Union[int, Tuple[int]] = 224,
                 share_embedding: bool = True,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.share_embedding = share_embedding

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding (not needed if only masked tokens are given as input, but can be useful to train Token Critic)
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens)

        # Output projection layer
        self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

        if self.share_embedding:
            # Share input and output embedding weights
            self.to_logits.weight = self.token_emb.weight

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the embedding module, transforming tokenized spatial inputs to embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict, with at least the following key:
                - 'tensor' (torch.Tensor): Modality tokens for each batch (e.g. from tokenized images). Shape (B, H, W) where B is the batch size, H and W are height and width after tokenization.


        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence, which is replaced by mask tokens in the 4M decoder. Shape (B, H*W, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the token sequence. Shape (B, H*W, D).
                - 'ids' (torch.Tensor): Reshaped token sequence from input dict, flattened in the spatial dimensions. Shape (B, H*W).
        """
        ids = d['tensor']
        B = ids.shape[0]
        ids = ids.reshape(B, -1)

        # Map to embedding
        x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x
        d['emb'] = x_emb
        d['ids'] = ids
        return d

    @torch.compiler.disable(recursive=True)
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)
        
        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, M, V)
        """
        logits = self.to_logits(x)
        return logits


class ContinuousImageTokenDecoderEmbedding(nn.Module):
    """Embedding module for continuous tokenized spatial inputs.

    Args:
        data_dim: dimensionality of data
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(
        self,
        data_dim: int,
        flow: Literal["nsf", "rf"] = "rf",
        conditioning_dim: int = 256,
        patch_size: Union[int, Tuple[int, int]] = 16,
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: bool = True,
        image_size: Union[int, Tuple[int]] = 224,
        share_embedding: bool = False,  # not used
        # for rectified flow
        rf_schedule_alpha: float = 2.0,
        rf_schedule_beta: float = 2.0,
        rf_mlp_ratio: float = 2.0,
        rf_mlp_depth: int = 2,
        rf_dropout: float = 0.0,
        # for nsf
        nsf_layers: int = 3,
        nsf_bins: int = 8,
        nsf_hidden_dims: list[int] = [256, 256],
        rescale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.flow_class = flow
        self.conditioning_dim = conditioning_dim
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )
        self.share_embedding = share_embedding

        if self.flow_class == "rf":
            self.schedule_alpha = rf_schedule_alpha
            self.schedule_beta = rf_schedule_beta
            self.hidden_dim = int(self.conditioning_dim * rf_mlp_ratio)
            self.num_mlp_layers = rf_mlp_depth
            self.dropout = rf_dropout
        elif self.flow_class == "nsf":
            self.flow_layers = nsf_layers
            self.flow_bins = nsf_bins
            self.hidden_dim = nsf_hidden_dims
            self.rescale = rescale
        else:
            raise ValueError(
                f"Flow class {self.flow_class} not supported, choose 'rf' or 'nsf'"
            )

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(
                h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens
            )
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens)
            )
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # # Token embedding (not needed if only masked tokens are given as input, but can be useful to train Token Critic)
        # self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens)
        self.token_emb = nn.Linear(self.data_dim, self.dim_tokens)

        # Output projection layer
        # self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)
        self.to_conditioner = nn.Linear(self.dim_tokens, self.conditioning_dim)
        if self.flow_class == "rf":
            self.flow = RectifiedFlow(
                data_dim=self.data_dim,
                time_schedule=BetaSchedule(
                    alpha=self.schedule_alpha, beta=self.schedule_beta
                ),
                score_model=MLPBackbone(
                    input_dim=self.data_dim,
                    cond_dim=self.conditioning_dim,
                    embed_dim=self.conditioning_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_mlp_layers,
                    activation="SiLU",
                    dropout=self.dropout,
                ),
            )
        elif self.flow_class == "nsf":
            self.flow = NSF(
                data_dim=self.data_dim,
                context_dim=self.conditioning_dim,
                layers=self.flow_layers,
                bins=self.flow_bins,
                hidden_dims=self.hidden_dim,
                activation="GELU",
                rescale=self.rescale,
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the embedding module, transforming tokenized spatial inputs to embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict, with at least the following key:
                - 'tensor' (torch.Tensor): Modality tokens for each batch (e.g. from tokenized images). Shape (B, H, W, C) where B is the batch size, H and W are height and width after tokenization.


        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence, which is replaced by mask tokens in the 4M decoder. Shape (B, H*W, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the token sequence. Shape (B, H*W, D).
                - 'ids' (torch.Tensor): Reshaped token sequence from input dict, flattened in the spatial dimensions. Shape (B, H*W, C).
        """
        ids = d["tensor"]
        B, *spatial_dims, C = ids.shape
        ids = ids.reshape(B, -1, C)

        # Map to embedding
        x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = ids
        return d

    @torch.compiler.disable(recursive=True)
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to flow conditioners.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, N, D)

        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, N, C)
        """
        conditioner = self.to_conditioner(x)
        return conditioner

    def forward_loss(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Computes loss for the flow model.

        Args:
            x (torch.Tensor): Tokens to calculate the likelihood for. Shape (N, D)
            ctx (torch.Tensor): Context for the flow model. Shape (N, C)

        Returns:
            torch.Tensor: per-item loss. Shape (N)
        """
        # b, n, d = x.shape
        # _loss = self.flow.loss(x.reshape(b * n, -1), context=ctx.reshape(b * n, -1)).reshape(b, n)
        # return _loss
        return self.flow.loss(x, context=ctx)

    def sample(self, ctx: torch.Tensor, temperature: float = 1.0, atol=1e-5, rtol=1e-5):
        """
        Sample from the flow model.

        Args:
            ctx (torch.Tensor): Context for the flow model. Shape (N, C)
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Samples from the flow model. Shape (N, num_samples, D)
        """
        return self.flow.sample(ctx, temperature=temperature, atol=atol, rtol=rtol)
