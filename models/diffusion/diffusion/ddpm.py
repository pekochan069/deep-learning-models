from typing import final, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from core.config import DiffusionConfig
from core.weights import load_model, save_model
from ..base_model import DiffusionBaseModel


@final
class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.1):
        super(ResBlock, self).__init__()

        num_groups = min(32, out_dim // 4)

        self.sequential = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, in_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, in_dim),
            nn.SiLU(),
        )

        if in_dim != out_dim:
            self.residual_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)
        else:
            self.residual_conv = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _ = nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(m, nn.GroupNorm):
                _ = nn.init.ones_(m.weight)
                _ = nn.init.zeros_(m.bias)

    @override
    def forward(self, x: torch.Tensor):
        o = self.sequential(x)

        return o + self.residual_conv(x)


@final
class DownBlock(nn.Module):
    def __init__(self, in_dim: int):
        super(DownBlock, self).__init__()

        out_dim = in_dim * 2
        num_groups = min(32, out_dim // 4)

        self.sequential = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 2, 1, bias=False),
            nn.GroupNorm(num_groups, out_dim),
            nn.SiLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _ = nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(m, nn.GroupNorm):
                _ = nn.init.ones_(m.weight)
                _ = nn.init.zeros_(m.bias)

    @override
    def forward(self, x: torch.Tensor):
        o = self.sequential(x)

        return o


@final
class UpBlock(nn.Module):
    def __init__(self, in_dim: int):
        super(UpBlock, self).__init__()

        out_dim = in_dim // 2
        num_groups = min(32, out_dim // 4)

        self.sequential = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups, out_dim),
            nn.SiLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _ = nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(m, nn.GroupNorm):
                _ = nn.init.ones_(m.weight)
                _ = nn.init.zeros_(m.bias)

    @override
    def forward(self, x: torch.Tensor):
        o = self.sequential(x)

        return o


@final
class UNet(nn.Module):
    def __init__(self, in_dim: int):
        super(UNet, self).__init__()

        self.block1 = nn.Sequential(
            ResBlock(in_dim, 64),
            ResBlock(64, 64),
        )
        self.block2 = nn.Sequential(
            DownBlock(64),
            ResBlock(128, 128),
            ResBlock(128, 128),
        )
        self.block3 = nn.Sequential(
            DownBlock(128),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.block4 = nn.Sequential(
            DownBlock(256),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )

        self.up1 = UpBlock(512)
        self.block5 = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            UpBlock(256),
        )
        self.block6 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            UpBlock(128),
        )
        self.block7 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, in_dim, 1, 1, 0, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _ = nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(m, nn.GroupNorm):
                _ = nn.init.ones_(m.weight)
                _ = nn.init.zeros_(m.bias)

    @override
    def forward(self, x: torch.Tensor):
        o1 = self.block1(x)
        o2 = self.block2(o1)
        o3 = self.block3(o2)
        o4 = self.block4(o3)

        o = self.up1(o4)
        o = o3 + o
        o = self.block5(o)
        o = o2 + o
        o = self.block6(o)
        o = o1 + o
        o = self.block7(o)


@final
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self):
        super(SinusoidalPositionEmbedding, self).__init__()

    @override
    def forward(self, x: torch.Tensor):
        return x


@final
class DDPM(DiffusionBaseModel):
    """DDPM

    DDPM With CFG
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)

    @override
    def forward(self, x: torch.Tensor):
        return x

    @override
    def save(self):
        save_model(self, f"{self.config.name}")

    @override
    def load(self):
        loaded_model = load_model(self, f"{self.config.name}")

        if loaded_model is None:
            self.logger.info(f"Model {self.config.name} not found.")
            return

        _ = self.load_state_dict(loaded_model.state_dict())
        self.logger.info(f"Model {self.config.name} loaded successfully.")

    @override
    def to_cpu(self):
        _ = self.to(self.device_cpu)

    @override
    def to_device(self):
        _ = self.to(self.device)

    @override
    def train_epoch(self, *args, **kwargs):
        return super().train_epoch(*args, **kwargs)

    @override
    def validate_epoch(self, *args, **kwargs):
        return super().validate_epoch(*args, **kwargs)

    @override
    def fit(self, *args, **kwargs) -> None:
        return super().fit(*args, **kwargs)

    @override
    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)

    @override
    def summary(self, *args, **kwargs):
        return super().summary(*args, **kwargs)

    @override
    def plot_history(self, show: bool, save: bool):
        return super().plot_history(show, save)
