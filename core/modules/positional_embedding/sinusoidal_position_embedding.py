# pyright: reportIndexIssue=false, reportCallIssue=false

from typing import final, override

import numpy as np
import torch
import torch.nn as nn


@final
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d: int,
        max_len: int = 10000,
        base: float = 10000,
    ):
        super(SinusoidalPositionalEmbedding, self).__init__()
        assert d % 2 == 0

        self.d = d

        pe = torch.zeros(max_len, d)
        p = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, d, 2, dtype=torch.float32)
        term = torch.exp((-np.log(base) / d) * i)

        x = p * term

        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)

        # pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe, persistent=False)

    # made by chatgpt
    @override
    def forward(self, t: torch.Tensor):
        # return self.pe[:seq_len, :]
        if t.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        ):
            index = t.long().flatten()

            # numel - 요소 개수
            if index.numel() > 0 and index.max() < self.pe.size(0) and index.min() >= 0:
                embedding = self.pe.index_select(0, index)
                return embedding.view(*t.shape, self.d)

        t = t.to(dtype=torch.float32)
        orig_shape = t.shape
        t = t.view(-1)  # (N,)

        half = self.d // 2
        # freq = base^{-k/half}, k=0..half-1
        k = torch.arange(0, half, device=t.device, dtype=t.dtype)
        freqs = torch.exp(
            -torch.log(torch.tensor(self.base, device=t.device, dtype=t.dtype))
            * (k / half)
        )
        args = t[:, None] * freqs[None, :]  # (N, half)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, d)
        return emb.view(*orig_shape, self.d)


@final
class SinusoidalPositionalEmbedding2d(nn.Module):
    def __init__(self, image_size: int, base: float = 10000):
        super(SinusoidalPositionalEmbedding2d, self).__init__()

        self.image_size = image_size
        self.log_base = np.log(base)

    @override
    def forward(self, d: int):
        assert d % 4 == 0
        d_h = d_w = d // 2

        pe_h = torch.zeros(self.image_size, d_h)
        p_h = torch.arange(0, self.image_size, dtype=torch.float32).unsqueeze(1)
        i_h = torch.arange(0, d_h, 2, dtype=torch.float32)
        term_h = torch.exp((-self.log_base / d_h) * i_h)

        x_h = p_h * term_h

        pe_h[:, 0::2] = torch.sin(x_h)
        pe_h[:, 1::2] = torch.cos(x_h)

        pe_w = torch.zeros(self.image_size, d_w)
        p_w = torch.arange(0, self.image_size, dtype=torch.float32).unsqueeze(1)
        i_w = torch.arange(0, d_w, 2, dtype=torch.float32)
        term_w = torch.exp((-self.log_base / d_w) * i_w)

        x_w = p_w * term_w

        pe_w[:, 0::2] = torch.sin(x_w)
        pe_w[:, 1::2] = torch.cos(x_w)

        pe = torch.cat(
            [
                pe_w.unsqueeze(0).repeat(self.image_size, 1, 1),
                pe_h.unsqueeze(1).repeat(1, self.image_size, 1),
            ],
            dim=-1,
        )

        return pe
