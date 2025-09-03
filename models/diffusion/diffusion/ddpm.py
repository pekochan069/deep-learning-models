from typing import final, override

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import DiffusionConfig
from core.dataset import get_dataset_info
from core.modules import SinusoidalPositionalEmbedding
from core.weights import load_model, save_model
from ..base_model import DiffusionBaseModel

_log_10000 = np.log(10000)


def sinusoidal_positional_embedding(p: int, i: int, d: int, device: torch.device):
    x = p * np.exp((-2 * i / d) * _log_10000)

    if i % 2 == 0:
        y = np.sin(x)
    else:
        y = np.cos(x)

    return torch.Tensor(y, device=device)


@final
class SelfAttention(nn.Module):
    def __init__(self, in_dim: int):
        super(SelfAttention, self).__init__()

    @override
    def forward(self, x: torch.Tensor):
        return x


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
            SelfAttention(128),
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
            SelfAttention(128),
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
    def forward(self, x: torch.Tensor, t: int, y: torch.Tensor):
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
class DDPM(DiffusionBaseModel):
    """DDPM

    DDPM With CFG
    """

    def __init__(
        self, config: DiffusionConfig, max_t: int, beta_1: float, beta_t: float
    ):
        assert max_t > 0
        assert 0 < beta_1 < 1
        assert 0 < beta_t < 1
        super().__init__(config)

        self.dataset_info = get_dataset_info(self.config.dataset)

        self.max_t = max_t
        self.beta_1 = beta_1
        self.beta_t = beta_t
        self.alpha_t = 1 - self.beta_t

        self.betas = torch.linspace(self.beta_1, self.beta_t, self.max_t)
        self.alphas = self.betas - 1
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)

        self.unet = UNet(self.dataset_info.channels)

    @override
    def forward(self, x: torch.Tensor):
        return x

    @override
    def save(self):
        save_model(self.unet, f"{self.config.name}-unet")

    @override
    def load(self):
        loaded_model = load_model(self.unet, f"{self.config.name}-unet")

        if loaded_model is None:
            self.logger.info(f"Model {self.config.name} not found.")
            return

        _ = self.unet.load_state_dict(loaded_model.state_dict())
        self.logger.info(f"Model {self.config.name} loaded successfully.")

    @override
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ):
        _ = self.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            x, y = batch

            x = x.to(self.device)
            y = y.to(self.device)

            batch_size = x.size(0)

            t = torch.randint(0, self.max_t, (batch_size,), device=self.device)
            eps = torch.randn(x.shape, device=self.device)

            sqrt_alphas_bar = self.sqrt_alphas_bar[t].view(batch_size, 1, 1, 1)
            sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[t].view(
                batch_size, 1, 1, 1
            )

            x_t = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar * eps

            eps_t = self.unet(x_t, t, y)

            loss = loss_function(eps, eps_t)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss

    @override
    def validate_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        loss_function: nn.Module,
    ):
        _ = self.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                batch_size = x.size(0)

                t = torch.randint(0, self.max_t, (batch_size,))
                eps = torch.randn(x.shape, device=self.device)

                sqrt_alphas_bar = self.sqrt_alphas_bar[t].view(batch_size, 1, 1, 1)
                sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[t].view(
                    batch_size, 1, 1, 1
                )

                x_t = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar * eps

                eps_t = self.unet(x_t, t, y)

                loss = loss_function(eps, eps_t)

                epoch_loss += loss

        return epoch_loss

    @override
    def fit(self, *args, **kwargs) -> None:
        return super().fit(*args, **kwargs)

    @override
    def predict(
        self, batch_size: int = 64, steps: int = 20, guidance_scale: float = 2.0
    ):
        _ = self.eval()

        with torch.no_grad():
            x_t = torch.randn(
                (
                    batch_size,
                    self.dataset_info.channels,
                    self.dataset_info.image_size,
                    self.dataset_info.image_size,
                ),
                device=self.device,
            )

            for i in tqdm(range(1, steps + 1), desc="Generating"):
                if i != steps:
                    z = torch.randn(
                        (
                            batch_size,
                            self.dataset_info.channels,
                            self.dataset_info.image_size,
                            self.dataset_info.image_size,
                        ),
                        device=self.device,
                    )
                else:
                    z = torch.zeros(
                        (
                            batch_size,
                            self.dataset_info.channels,
                            self.dataset_info.image_size,
                            self.dataset_info.image_size,
                        ),
                        device=self.device,
                    )

                alphas = self.alphas[i + 1].view(batch_size, 1, 1, 1)
                alphas_bar = self.alphas_bar[i + 1].view(batch_size, 1, 1, 1)
                sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[i + 1].view(
                    batch_size, 1, 1, 1
                )
                betas = self.betas[i + 1].view(batch_size, 1, 1, 1)

                x_t = (1 / alphas_bar) * (
                    x_t - ((1 - alphas) / (sqrt_one_minus_alphas_bar)) * self.unet(x_t)
                ) + z * betas

    @override
    def summary(self, *args, **kwargs):
        return super().summary(*args, **kwargs)

    @override
    def plot_history(self, show: bool, save: bool):
        return super().plot_history(show, save)
