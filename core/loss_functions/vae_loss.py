from typing import final, override

import torch
import torch.nn as nn


@final
class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

        self.mse = nn.MSELoss(reduction="sum")

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        l1 = self.mse(x, x_hat)
        l2 = torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)

        return (l1 - l2) / (2 * len(x))
