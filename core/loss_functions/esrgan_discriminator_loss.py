from typing import final, override
import torch
import torch.nn as nn
import torch.nn.functional as F


@final
class ESRGANDiscriminatorLoss(nn.Module):
    fake_label: float
    real_label: float
    d_x_loss_function: nn.Module
    d_g_z_loss_function: nn.Module

    def __init__(self, fake_label: float = 0.0, real_label: float = 1.0):
        super(ESRGANDiscriminatorLoss, self).__init__()

        self.fake_label = fake_label
        self.real_label = real_label
        self.d_x_loss_function = nn.BCELoss()
        self.d_g_z_loss_function = nn.BCELoss()

    @override
    def forward(
        self,
        d_x: torch.Tensor,
        d_g_z: torch.Tensor,
    ) -> torch.Tensor:
        d_ra_real = F.sigmoid(d_x - torch.mean(d_g_z))
        d_ra_fake = F.sigmoid(d_g_z - torch.mean(d_x))
        d_real_loss = self.d_x_loss_function(
            d_ra_real, torch.full_like(d_ra_real, self.real_label)
        )
        d_fake_loss = self.d_g_z_loss_function(
            d_ra_fake, torch.full_like(d_ra_fake, self.fake_label)
        )

        return (d_real_loss + d_fake_loss) / 2
