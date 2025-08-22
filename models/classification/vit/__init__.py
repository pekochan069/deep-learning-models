from typing import final, override

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from core.config import ClassificationConfig


@final
class Attention(nn.Module):
    def __init__(self, *, in_channels: int, heads: int, use_mask: bool):
        super(Attention, self).__init__()

        assert in_channels % heads == 0

        self.heads = heads
        self.head_dim = int(heads / in_channels)
        self.mask = use_mask

        self.q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.k = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.v = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.softmax = nn.Softmax(dim=-1)

        self.proj_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        q = self.q(x)  # b c h w
        k = self.k(x)  # b c h w
        v = self.v(x)  # b c h w

        q = rearrange(
            q, "b (heads d) h w -> b heads (h w) d", heads=self.heads
        )  # b heads hw head_dim
        k = rearrange(
            k, "b (heads d) h w -> b heads (h w) d", heads=self.heads
        )  # b heads hw head_dim
        v = rearrange(
            v, "b (heads d) h w -> b heads (h w) d", heads=self.heads
        )  # b heads hw head_dim

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(
            self.head_dim
        )  # b heads hw_q hw_k

        if self.mask:
            pass

        o = self.softmax(attn)

        o = torch.einsum("bhqk,bhvd->bhqd", attn, v)  # b heads head_dim hw_k

        o = rearrange(o, "b heads (h w) d -> b (heads d) h w", h=h, w=w)

        o = self.proj_out(o) + x

        return o


@final
class LinearAttention(nn.Module):
    def __init__(self, in_channels: int, heads: int, head_dim: int):
        super(LinearAttention, self).__init__()

    @override
    def forward(self, x: torch.Tensor):
        return x


@final
class ViT(nn.Module):
    def __init__(self, config: ClassificationConfig) -> None:
        super(ViT, self).__init__()
