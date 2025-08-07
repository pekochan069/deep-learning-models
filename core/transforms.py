from typing import final, override

import torch
import torch.nn as nn


@final
class RealESEGANBlur(nn.Module):
    def __init__(self):
        super(RealESEGANBlur, self).__init__()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = x

        return o
