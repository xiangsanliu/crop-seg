import logging
import math
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
import math

from tools.helper import load_pretained


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.embed_dim,
                              kernel_size=config.patch_size, stride=config.patch_size[0])
        # self.act = nn.GELU()
        self.config = config
        # self.layer_norm = nn.LayerNorm(
        # normalized_shape=config.embed_dim, eps=config.layer_norm_eps)

    def forward(self, x):
        B, C, H, W = x.shape
        assert self.config.in_channels == C, ' in_channels != 输入图像通道'
        x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.act(x)
        # x = self.layer_norm(x)
        return x


class PositionEmbedding(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim*3)
        self.num_heads = config.num_heads
        head_dim = config.embed_dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(config.attention_drop_ratio)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj_drop = nn.Dropout(config.proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(config.drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = Attention(config)
        self.drop_path = DropPath(
            config.drop_path_ratio) if config.drop_path_ratio > 0 else nn.Identity()
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = Mlp(config, in_features=config.embed_dim,
                       hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.layer_norm1(x)))
        x = x + self.drop_path(self.mlp(self.layer_norm2(x)))
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v

        self.patch_embedding = PatchEmbedding(config)
        # self.position_embeddings = PositionEmbedding(config)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, config.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.drop_ratio)
        self.blocks = nn.Sequential(*[
            Block(config) for i in range(config.num_hidden_layers)
        ])
        self.blocks = nn.ModuleList([Block(config)
                                     for _ in range(config.num_hidden_layers)])
        # self.fc = nn.Linear(
        # config.embed_dim, config.patch_size[0]*config.patch_size[1]*config.embed_dim//(sample_v**2))

        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.init_weights(config)

    def forward(self, x):
        b, c, h, w = x.shape
        B = x.shape[0]
        x = self.patch_embedding(x)
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # x = self.position_embeddings(x)
        outs = []
        for _, block in enumerate(self.blocks):
            layer_output = block(x)
            x = layer_output
            outs.append(x)
        encode_x = outs[-1]
        # x = self.fc(encode_x)
        # hh = h // self.patch_size[0]
        # ww = w // self.patch_size[1]
        # print(x.shape)
        # x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        #   p1=self.hh, p2=self.ww, h=hh, w=ww, c=self.config.embed_dim)
        return outs[-1]

    def init_weights(self, config):
        load_pretained(self, config)
