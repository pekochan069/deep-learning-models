from typing import Literal, final, override

from pydantic import ConfigDict
import torch
import torch.nn as nn

from core.pydantic import ParametersBase


@final
class VQVAELoss(nn.Module):
    def __init__(self, beta: float = 0.25):
        super(VQVAELoss, self).__init__()

        self.mse = nn.MSELoss()

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        q_loss: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_loss = self.mse(x, x_hat)

        return reconstruction_loss + q_loss


class VQVAELossParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    param_type: Literal["vq_vae_loss"] = "vq_vae_loss"
