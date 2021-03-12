from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .model import PretrainedModel


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(PretrainedModel):
    def __init__(self, config, img_dim, input_resolution):
        super().__init__(config)
        self.input_resolution = input_resolution
        self.output_dim = img_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=config.vit_hidden_size,
                               kernel_size=config.patch_size, stride=config.patch_size, bias=False)

        scale = config.vit_hidden_size ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(config.vit_hidden_size))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // config.patch_size) ** 2 + 1,
                                config.vit_hidden_size))
        self.ln_pre = LayerNorm(config.vit_hidden_size)

        self.transformer = Transformer(config.vit_hidden_size, config.vit_num_layers,
                                       config.vit_num_heads)

        self.ln_post = LayerNorm(config.vit_hidden_size)
        self.proj = nn.Parameter(scale * torch.randn(config.vit_hidden_size, img_dim))

    def forward(self, image_tensor: torch.Tensor):
        # Convert raw image tensor to patch embeddings for each grid
        patch_emb = self.conv1(image_tensor)  # shape = [*, width, grid, grid]
        patch_emb = patch_emb.reshape(patch_emb.shape[0], patch_emb.shape[1], -1)  # shape = [*, width, grid ** 2]
        patch_emb = patch_emb.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Construct sequence embedding with CLS token and positional embedding
        seq_emb = torch.cat([
            self.class_embedding.to(patch_emb.dtype) + torch.zeros(
                patch_emb.shape[0], 1, patch_emb.shape[-1], dtype=patch_emb.dtype,
                device=patch_emb.device), patch_emb], dim=1)  # shape = [*, grid ** 2 + 1, width]
        seq_emb = seq_emb + self.positional_embedding.to(seq_emb.dtype)
        seq_emb = self.ln_pre(seq_emb)
        seq_emb = seq_emb.permute(1, 0, 2)  # NLD -> LND

        # Apply transformer layers to embedding input
        encoder_output = self.transformer(seq_emb)
        encoder_output = encoder_output.permute(1, 0, 2)  # LND -> NLD
        output = self.ln_post(encoder_output[:, 0, :])

        # Linear/MLP projection for CLS
        if self.proj is not None:
            output = output @ self.proj

        return output
