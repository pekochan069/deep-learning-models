# pyright: reportIndexIssue=false

from typing import Literal, final, override

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import DiffusionConfig
from core.dataset import get_dataset_info
from core.modules.positional_embedding import SinusoidalPositionalEmbedding
from core.modules.attention import MultiHeadScaledDotAttention
from core.weights import load_model, save_model
from ..base_model import DiffusionBaseModel


def gn_groups(c: int, max_groups: int = 32) -> int:
    g = np.gcd(c, max_groups)
    return max(1, g)


@final
class ResBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        t_emb_dim: int,
        dropout_rate: float,
    ):
        super(ResBlock, self).__init__()
        assert in_dim > 3

        num_groups = gn_groups(in_dim)

        self.embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_dim, bias=False),
        )

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_dim),
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
        num_groups = gn_groups(in_dim)

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
    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        skip_type: Literal["add", "concat"],
        skip_reduce: float,
    ):
        super(UpBlock, self).__init__()

        out_dim = in_dim // 2
        num_groups = gn_groups(in_dim)

        self.skip_type: Literal["add", "concat"] = skip_type

        # self.upsample = nn.Upsample(scale_factor=2)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.GroupNorm(num_groups, in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
        )

        if self.skip_type == "concat":
            skip_reduced_dim = max(1, int(round(skip_dim * skip_reduce)))
        else:
            skip_reduced_dim = out_dim
        self.skip_conv = nn.Conv2d(skip_dim, skip_reduced_dim, 1, bias=False)

        self.concat_conv = nn.Sequential(
            nn.GroupNorm(num_groups, out_dim + skip_reduced_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim + skip_reduced_dim, out_dim, 3, 1, 1, bias=False),
        )

        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
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
    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        o = self.upsample(x)

        if o.shape[-2:] != skip.shape[-2:]:
            o = F.interpolate(o, size=skip.shape[-2:])

        skip = self.skip_conv(skip)

        if self.skip_type == "concat":
            o = torch.cat([o, skip], dim=1)
            o = self.concat_conv(o)
        else:
            # 안정화를 위한 1/√2 스케일
            o = (o + skip) * 0.70710678

        o = self.conv(o)

        return o


@final
class UNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        base_unet_dim: int,
        t_dim: int,
        num_classes: int,
        dropout: float,
        skip_type: Literal["add", "concat"],
        skip_reduce: float,
    ):
        super(UNet, self).__init__()

        num_groups = gn_groups(in_dim)

        unet_dim_x2 = base_unet_dim * 2
        unet_dim_x4 = base_unet_dim * 4
        unet_dim_x8 = base_unet_dim * 8

        t_emb_dim = t_dim * 4

        self.time_embedding = SinusoidalPositionalEmbedding(t_dim)
        self.class_embedding = nn.Embedding(num_classes + 1, t_emb_dim)
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(t_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        self.block0 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_dim, base_unet_dim, 3, 1, 1, bias=False),
        )

        self.block1_1 = ResBlock(
            base_unet_dim,
            base_unet_dim,
            t_emb_dim,
            dropout,
        )
        self.block1_2 = ResBlock(
            base_unet_dim,
            base_unet_dim,
            t_emb_dim,
            dropout,
        )

        self.down1 = DownBlock(base_unet_dim)

        self.block2_1 = ResBlock(
            unet_dim_x2,
            unet_dim_x2,
            t_emb_dim,
            dropout,
        )
        self.block2_attention = MultiHeadScaledDotAttention(unet_dim_x2, 32, 8, 0.1)
        self.block2_2 = ResBlock(
            unet_dim_x2,
            unet_dim_x2,
            t_emb_dim,
            dropout,
        )

        self.down2 = DownBlock(unet_dim_x2)

        self.block3_1 = ResBlock(
            unet_dim_x4,
            unet_dim_x4,
            t_emb_dim,
            dropout,
        )
        self.block3_2 = ResBlock(
            unet_dim_x4,
            unet_dim_x4,
            t_emb_dim,
            dropout,
        )

        self.down3 = DownBlock(unet_dim_x4)

        self.block4_1 = ResBlock(
            unet_dim_x8,
            unet_dim_x8,
            t_emb_dim,
            dropout,
        )
        self.block4_2 = ResBlock(
            unet_dim_x8,
            unet_dim_x8,
            t_emb_dim,
            dropout,
        )

        self.up1 = UpBlock(unet_dim_x8, unet_dim_x4, skip_type, skip_reduce)

        self.block5_1 = ResBlock(
            unet_dim_x4,
            unet_dim_x4,
            t_emb_dim,
            dropout,
        )
        self.block5_2 = ResBlock(
            unet_dim_x4,
            unet_dim_x4,
            t_emb_dim,
            dropout,
        )

        self.up2 = UpBlock(unet_dim_x4, unet_dim_x2, skip_type, skip_reduce)

        self.block6_1 = ResBlock(
            unet_dim_x2,
            unet_dim_x2,
            t_emb_dim,
            dropout,
        )
        self.block6_attention = MultiHeadScaledDotAttention(unet_dim_x2, 32, 8, 0.1)
        self.block6_2 = ResBlock(
            unet_dim_x2,
            unet_dim_x2,
            t_emb_dim,
            dropout,
        )

        self.up3 = UpBlock(unet_dim_x2, base_unet_dim, skip_type, skip_reduce)

        self.block7_1 = ResBlock(
            base_unet_dim,
            base_unet_dim,
            t_emb_dim,
            dropout,
        )
        self.block7_2 = ResBlock(
            base_unet_dim,
            base_unet_dim,
            t_emb_dim,
            dropout,
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups, base_unet_dim),
            nn.SiLU(),
            nn.Conv2d(base_unet_dim, in_dim, 3, 1, 1, bias=False),
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
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None):
        t_emb = self.time_embedding(t)
        t_emb = self.time_embedding_mlp(t_emb)
        if y is not None:
            class_emb = self.class_embedding(y)
            t_emb += class_emb

        o0 = self.block0(x)

        o1 = self.block1_1(o0, t_emb)
        o1 = self.block1_2(o1, t_emb)
        o1 = self.down1(o1)
        o2 = self.block2_1(o1, t_emb)
        o2 = self.block2_attention(o2)
        o2 = self.block2_2(o2, t_emb)
        o2 = self.down2(o2)
        o3 = self.block3_1(o2, t_emb)
        o3 = self.block3_2(o3, t_emb)
        o3 = self.down3(o3)
        o4 = self.block4_1(o3, t_emb)
        o4 = self.block4_2(o4, t_emb)

        o = self.up1(o4, o3)
        o = self.block5_1(o, t_emb)
        o = self.block5_2(o, t_emb)
        o = self.up2(o, o2)
        o = self.block6_1(o, t_emb)
        o = self.block6_attention(o)
        o = self.block6_2(o, t_emb)
        o = self.up3(o, o1)
        o = self.block7_1(o, t_emb)
        o = self.block7_2(o, t_emb)

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
        base_unet_dim: int,
        t_emb_dim: int,
        dropout: float,
        long_skip_connection_type: Literal["add", "concat"],
        skip_reduce_ratio: float,
        gradient_clipping: bool,
        max_clip_norm: int,
    ):
        assert max_t > 0
        assert 0 < beta_1 < 1
        assert 0 < beta_t < 1
        assert t_emb_dim > 0 and t_emb_dim % 4 == 0
        super().__init__(config)

        self.dataset_info = get_dataset_info(self.config.dataset)

        self.max_t = max_t
        self.beta_1 = beta_1
        self.beta_t = beta_t
        self.alpha_t = 1 - self.beta_t
        self.cfg_unconditional_prob = cfg_unconditional_prob
        self.t_emb_dim = t_emb_dim
        self.null_token = self.dataset_info.num_classes
        self.gradient_clipping = gradient_clipping
        self.max_norm = max_clip_norm

        betas = torch.linspace(self.beta_1, self.beta_t, self.max_t)
        alphas: torch.Tensor = 1 - betas
        alphas_bar = torch.cumprod(alphas, -1)
        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
        self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)

        self.unet = UNet(
            self.dataset_info.channels,
            base_unet_dim,
            t_emb_dim,
            self.dataset_info.num_classes,
            dropout,
            long_skip_connection_type,
            skip_reduce_ratio,
        )

    def apply_cfg_conditioning(self, y: torch.Tensor) -> torch.Tensor:
        probs = torch.rand(y.shape[0], device=self.device)
        mask = probs < self.cfg_unconditional_prob
        new_y = y.clone()
        new_y[mask] = self.null_token

        return new_y

    @override
    def forward(self, x: torch.Tensor):
        return x

    @override
    def save(
        self,
        label: str = "last",
        epoch: int | None = None,
        best_metric: float | None = None,
    ):
        # ema_sd = getattr(self, "ema", None)
        # ema_sd = ema_sd.state_dict() if hasattr(ema_sd, "state_dict") else None
        save_model(
            self.unet,
            self.config.name,
            f"unet-{label}",
            epoch=epoch,
            best_metric=best_metric,
            # ema_state_dict=ema_sd,
            optimizer=getattr(self, "optimizer", None),
            lr_scheduler=getattr(self, "lr_scheduler", None),
            scaler=getattr(self, "scaler", None),
            meta={"dataset": self.config.dataset},
        )

    @override
    def load(self, label: str = "last"):
        loaded_model = load_model(self.unet, self.config.name, label)

        if loaded_model is None:
            self.logger.info(f"Model {self.config.name} not found.")
            return

        # _ = self.unet.load_state_dict(loaded_model.state_dict())
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
            optimizer.zero_grad()
            x, y = batch

            x = x.to(self.device)
            y = y.to(self.device)
            y = self.apply_cfg_conditioning(y)

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
            if self.gradient_clipping:
                _ = nn.utils.clip_grad_norm_(self.unet.parameters(), self.max_norm)
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

                t = torch.randint(0, self.max_t, (batch_size,), device=self.device)
                eps = torch.randn(x.shape, device=self.device)

                sqrt_alphas_bar = self.sqrt_alphas_bar[t].view(batch_size, 1, 1, 1)
                sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[t].view(
                    batch_size, 1, 1, 1
                )

                x_t = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar * eps

                eps_t = self.unet(x_t, t, y)

                loss = loss_function(eps, eps_t)

                epoch_loss += loss.item()

        return epoch_loss

    @override
    def fit(self, *args, **kwargs) -> None:
        return super().fit(*args, **kwargs)

    @override
    def predict(
        self, batch_size: int = 64, steps: int = 20, guidance_scale: float = 2.0
    ):
        _ = self.eval()

        if steps >= self.max_t:
            self.logger.error(f"Steps must be smaller than max_t({self.max_t})")
            return

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
            t_space = torch.linspace(
                self.max_t - 1, 0, steps, dtype=torch.long, device=self.device
            )
            y = torch.randint(
                0,
                self.dataset_info.num_classes,
                (batch_size,),
                dtype=torch.long,
                device=self.device,
            )
            y_uncond = torch.full_like(y, self.null_token)

            for i in tqdm(range(0, steps), desc="Generating"):
                if i != steps - 1:
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

                sqrt_alphas_bar = self.sqrt_alphas_bar[i].view(batch_size, 1, 1, 1)
                sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[i].view(
                    batch_size, 1, 1, 1
                )
                betas = self.betas[i].view(batch_size, 1, 1, 1)

                t = t_space[i]

                o = self.unet(x_t, t, y)
                o_uncond = self.unet(x_t, t, y_uncond)
                o = o_uncond + guidance_scale * (o - o_uncond)

                x_t = (1 / sqrt_alphas_bar) * (
                    x_t - (betas / (sqrt_one_minus_alphas_bar)) * o
                ) + z * betas

        o = x_t.clamp(0, 1)
        o = o.cpu()

        grid_img = torchvision.utils.make_grid(o, nrow=8, padding=8, normalize=True)
        _ = plt.imshow(grid_img.permute(1, 2, 0))
        _ = plt.axis("off")
        _ = plt.show()

    @override
    def summary(self, *args, **kwargs):
        return super().summary(*args, **kwargs)

    @override
    def plot_history(self, show: bool, save: bool):
        return super().plot_history(show, save)
