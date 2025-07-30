from typing import final, override
import torch
import torch.nn as nn


@final
class GANDiscriminatorLoss(nn.Module):
    fake_label: float
    real_label: float
    d_x_loss_function: nn.Module
    d_g_z_loss_function: nn.Module

    def __init__(self, fake_label: float = 0.0, real_label: float = 1.0):
        super(GANDiscriminatorLoss, self).__init__()

        self.fake_label = fake_label
        self.real_label = real_label
        self.d_x_loss_function = nn.BCEWithLogitsLoss()
        self.d_g_z_loss_function = nn.BCEWithLogitsLoss()

    @override
    def forward(
        self,
        d_x: torch.Tensor,
        d_g_z: torch.Tensor,
    ) -> torch.Tensor:
        d_x_loss = self.d_x_loss_function(d_x, torch.full_like(d_x, self.real_label))
        d_g_z_loss = self.d_g_z_loss_function(
            d_g_z, torch.full_like(d_g_z, self.fake_label)
        )

        return (d_x_loss + d_g_z_loss) / 2
