from typing import final, override

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@final
class MultiHeadScaledDotAttention(nn.Module):
    def __init__(self, in_dim: int, head_dim: int, heads: int, dropout: float):
        super(MultiHeadScaledDotAttention, self).__init__()

        self.head_dim = head_dim
        self.heads = heads
        query_dim = head_dim * heads
        self.scale_factor = np.sqrt(head_dim)
        self.dropout = dropout

        self.pre_norm = nn.GroupNorm(min(32, in_dim // 4), in_dim)

        self.qkv = nn.Conv2d(in_dim, query_dim * 3, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.out_conv = nn.Conv2d(query_dim, in_dim, 1, bias=False)
        self.out_norm = nn.GroupNorm(min(32, in_dim // 4), in_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _ = nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    @override
    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape

        x = self.pre_norm(x)

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, 1)

        q = rearrange(q, "b (heads d) h w -> b heads (h w) d", heads=self.heads)
        k = rearrange(k, "b (heads d) h w -> b heads (h w) d", heads=self.heads)
        v = rearrange(v, "b (heads d) h w -> b heads (h w) d", heads=self.heads)

        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)

        o = rearrange(o, "b heads (h w) d -> b (heads d) h w", h=h, w=w)

        o = self.out_conv(o) + x
        o = self.out_norm(o)

        return o
