import logging
import math
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
import math


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channel, config.embed_dim,
                              kernel_size=config.patch_size, stride=config.patch_size[0])
        self.act = nn.GELU()
        self.config = config
        self.layer_norm = nn.LayerNorm(
            normalized_shape=config.embed_dim, eps=config.layer_norm_eps)

    def forward(self, x):
        B, C, H, W = x.shape
        assert self.config.in_channel == C, ' in_channels != 输入图像通道'
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.layer_norm(x)
        return x


class PositionEmbeding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            config.num_patches, config.embed_dim)
        self.layer_norm = nn.LayerNorm(
            config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        input_shape = x.shape
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape[:2])

        position_embeddings = self.position_embeddings(position_ids)
        x = x+position_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim*3)
        self.num_heads = config.num_heads
        head_dim = config.embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


