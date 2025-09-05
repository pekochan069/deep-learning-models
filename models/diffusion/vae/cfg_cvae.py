from typing import final, override

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from einops import repeat
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import DiffusionConfig
from core.dataset import get_num_classes
from core.registry import ModelRegistry
from .base_vae import VAEBaseModel


@final
class Encoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        in_dim: int,
        hidden_dim: int,
        hidden3_dim: int,
        latent_dim: int,
    ) -> None:
        super(Encoder, self).__init__()

        self.in_dim = in_dim

        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim + embedding_dim, hidden_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(
            hidden3_dim,
            latent_dim,
            bias=False,
        )
        self.fc_logvar = nn.Linear(
            hidden3_dim,
            latent_dim,
            bias=False,
        )

    @override
    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.embedding(y)
        y = repeat(y, "b y -> b y h w", h=x.size(2), w=x.size(3))
        x = torch.cat([x, y], dim=1)
        o = self.conv(x)
        o = self.flatten(o)
        mu = self.fc_mu(o)
        logvar = self.fc_logvar(o)
        sigma = torch.exp(logvar * 0.5)

        return mu, sigma


@final
class Decoder(nn.Module):
    def __init__(
        self,
        latent_image_size: int,
        num_classes: int,
        embedding_dim: int,
        latent_dim: int,
        hidden3_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super(Decoder, self).__init__()

        self.latent_image_size = latent_image_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_classes, embedding_dim)

        self.fc = nn.Linear(latent_dim + embedding_dim, hidden3_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim * 4,
                hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.embedding(y)
        x = torch.cat([x, y], dim=1)
        o = self.fc(x)
        o = o.view(
            -1, self.hidden_dim * 4, self.latent_image_size, self.latent_image_size
        )
        o = self.deconv(o)

        return o


@ModelRegistry.register("cfg_cvae")
@final
class CFGCVAE(VAEBaseModel):
    def __init__(
        self,
        config: DiffusionConfig,
        hidden_dim: int,
        latent_dim: int,
        embedding_dim: int,
    ):
        super().__init__(config)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        if self.config.dataset == "mnist":
            self.image_size = 28
            self.channels = 1
        elif self.config.dataset == "padded_mnist":
            self.image_size = 32
            self.channels = 1
        elif self.config.dataset == "cifar10" or self.config.dataset == "cifar100":
            self.image_size = 32
            self.channels = 3
        else:
            self.image_size = 224
            self.channels = 3
        self.num_classes = get_num_classes(self.config.dataset)
        self.latent_image_size = self.image_size // 8

        self.in_dim = self.channels
        self.hidden3_dim = (
            self.latent_image_size * self.latent_image_size * 4 * self.hidden_dim
        )

        self.encoder = Encoder(
            self.num_classes,
            self.embedding_dim,
            self.in_dim,
            self.hidden_dim,
            self.hidden3_dim,
            self.latent_dim,
        )
        self.decoder = Decoder(
            self.latent_image_size,
            self.num_classes,
            self.embedding_dim,
            self.latent_dim,
            self.hidden3_dim,
            self.hidden_dim,
            self.in_dim,
        )

    def apply_cfg_conditioning(
        self, y: torch.Tensor, uncond_prob: float, uncond_class: int
    ) -> torch.Tensor:
        probs = torch.rand(y.shape[0], device=self.device)
        mask = probs < uncond_prob
        new_y = y.clone()
        new_y[mask] = uncond_class

        return new_y

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

            y = self.apply_cfg_conditioning(y, 0.1, self.num_classes)

            optimizer.zero_grad()

            mu, sigma = self.encoder(x, y)
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps

            x_hat = self.decoder(z, y)

            loss = loss_function(x, x_hat, mu, sigma)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

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

                mu, sigma = self.encoder(x, y)
                eps = torch.randn_like(sigma)
                z = mu + sigma * eps

                x_hat = self.decoder(z, y)

                loss = loss_function(x, x_hat, mu, sigma)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    @override
    def predict(self, batch_size: int = 64, guidance_scale: float = 5.0):
        _ = self.eval()

        with torch.no_grad():
            z = torch.randn(
                (
                    batch_size,
                    self.latent_dim,
                ),
                device=self.device,
            )
            y = torch.randint(
                0, self.num_classes + 1, (batch_size,), device=self.device
            )
            y_uncond = self.apply_cfg_conditioning(y, 0.1, self.num_classes)
            o = self.decoder(z, y)
            o_uncond = self.decoder(z, y_uncond)
            o = o_uncond + guidance_scale * (o - o_uncond)
            o = o.clamp(0, 1)
            o = o.cpu()

        grid_img = torchvision.utils.make_grid(o, nrow=8, padding=8, normalize=True)
        _ = plt.imshow(grid_img.permute(1, 2, 0))
        _ = plt.axis("off")
        _ = plt.show()

    @override
    def summary(self, input_size: tuple[int, int, int, int]):
        pass
        # self.logger.info("Encoder summary")
        # _ = summary(self.encoder, input_size)
        # self.logger.info("Decoder summary")
        # _ = summary(
        #     self.decoder,
        #     (
        #         input_size[0],
        #         self.latent_dim,
        #     ),
        # )
