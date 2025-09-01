from typing import final, override, Literal

import torch
import torch.nn as nn
from pydantic import ConfigDict

from core.pydantic import ParametersBase


@final
class BetaVAELoss(nn.Module):
    def __init__(self, beta: float):
        super(BetaVAELoss, self).__init__()

        self.beta = beta

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
        l2 = self.beta * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)

        return (l1 - l2) / (2 * len(x))


class BetaVAELossParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    param_type: Literal["beta_vae_loss"] = "beta_vae_loss"
