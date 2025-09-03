# pyright: reportIndexIssue=false

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


@final
class SelfAttention(nn.Module):
    def __init__(self, in_dim: int):
        super(SelfAttention, self).__init__()

    @override
    def forward(self, x: torch.Tensor):
        return x


@final
class ResBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, t_emb_dim: int, dropout_rate: float = 0.1
    ):
        super(ResBlock, self).__init__()
        assert in_dim > 3

        self.embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_dim, bias=False),
        )

        self.conv1 = nn.Sequential(
            nn.GroupNorm(min(32, in_dim // 4), in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(32, out_dim // 4), out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
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
            elif isinstance(m, nn.Linear):
                _ = nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    @override
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        t_emb = self.embedding_projection(t_emb)

        o = self.conv1(x)
        o = o + t_emb[:, :, None, None]

        o = self.dropout(o)

        o = self.conv2(o)

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
    def __init__(self, in_dim: int, t_dim: int):
        super(UNet, self).__init__()

        t_emb_dim = t_dim * 4

        self.time_embedding = SinusoidalPositionalEmbedding(64)
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(t_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        self.block0 = nn.Sequential(
            nn.GroupNorm(1, in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, 64, 1, 1, 1, bias=False),
        )

        self.block1_1 = ResBlock(64, 64, 256)
        self.block1_2 = ResBlock(64, 64, 256)

        self.down1 = DownBlock(64)

        self.block2_1 = ResBlock(128, 128, 256)
        self.block2_2 = ResBlock(128, 128, 256)

        self.down2 = DownBlock(128)

        self.block3_1 = ResBlock(256, 256, 256)
        self.block3_2 = ResBlock(256, 256, 256)

        self.down3 = DownBlock(256)

        self.block4_1 = ResBlock(512, 512, 256)
        self.block4_2 = ResBlock(512, 512, 256)

        self.up1 = UpBlock(512)

        self.block5_1 = ResBlock(256, 256, 256)
        self.block5_2 = ResBlock(256, 256, 256)

        self.up2 = UpBlock(256)

        self.block6_1 = ResBlock(128, 128, 256)
        self.block6_2 = ResBlock(128, 128, 256)

        self.up3 = UpBlock(128)

        self.block7_1 = ResBlock(64, 64, 256)
        self.block7_2 = ResBlock(64, 64, 256)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, in_dim, 3, 1, 1, bias=False),
        )
        # self.block1 = nn.Sequential(
        #     ResBlock(in_dim, 64, 256),
        #     ResBlock(64, 64, 256),
        # )
        # self.block2 = nn.Sequential(
        #     DownBlock(64),
        #     ResBlock(128, 128, 256),
        #     SelfAttention(128),
        #     ResBlock(128, 128, 256),
        # )
        # self.block3 = nn.Sequential(
        #     DownBlock(128),
        #     ResBlock(256, 256, 256),
        #     ResBlock(256, 256, 256),
        # )
        # self.block4 = nn.Sequential(
        #     DownBlock(256),
        #     ResBlock(512, 512, 256),
        #     ResBlock(512, 512, 256),
        # )

        # self.block5 = nn.Sequential(
        #     ResBlock(256, 256, 256),
        #     ResBlock(256, 256, 256),
        #     UpBlock(256),
        # )
        # self.block6 = nn.Sequential(
        #     ResBlock(128, 128, 256),
        #     SelfAttention(128),
        #     ResBlock(128, 128, 256),
        #     UpBlock(128),
        # )
        # self.block7 = nn.Sequential(
        #     ResBlock(64, 64, 256),
        #     ResBlock(64, 64, 256),
        #     nn.Conv2d(64, in_dim, 1, 1, 0, bias=False),
        # )

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
        e = self.time_embedding(t)
        e += y

        o = self.block0(x)

        o1 = self.block1_1(o)
        o1 = self.block1_2(o1)
        o1 = self.down1(o1)
        o2 = self.block2_1(o1)
        o2 = self.block2_2(o2)
        o2 = self.down2(o2)
        o3 = self.block3_1(o2)
        o3 = self.block3_2(o3)
        o3 = self.down3(o3)
        o4 = self.block4_1(o3)
        o4 = self.block4_2(o4)

        o = self.up1(o4)
        o = o3 + o
        o = self.block5_1(o)
        o = self.block5_2(o)
        o = self.up2(o)
        o = o2 + o
        o = self.block6_1(o)
        o = self.block6_2(o)
        o = self.up3(o)
        o = o1 + o
        o = self.block7_1(o)
        o = self.block7_2(o)

        o = self.conv_out(o)

        return o


@final
class DDPM(DiffusionBaseModel):
    """DDPM

    DDPM With CFG
    """

    def __init__(
        self,
        config: DiffusionConfig,
        max_t: int,
        beta_1: float,
        beta_t: float,
        cfg_unconditional_prob: float,
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
        self.cfg_unconditional_prob = cfg_unconditional_prob

        betas = torch.linspace(self.beta_1, self.beta_t, self.max_t)
        alphas = betas - 1
        alphas_bar = torch.cumprod(alphas, 0)
        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
        self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)

        self.unet = UNet(self.dataset_info.channels)

    def apply_cfg_conditioning(self, y: torch.Tensor) -> torch.Tensor:
        probs = torch.rand(y.shape[0], device=self.device)
        mask = probs < self.cfg_unconditional_prob
        new_y = y.clone()
        new_y[mask] = self.dataset_info.num_classes + 1

        return new_y

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

            sqrt_alphas_bar = self.sqrt_alphas_bar[t].view(batch_size, 1, 1, 1)  #
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
