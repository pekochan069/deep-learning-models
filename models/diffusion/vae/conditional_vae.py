from typing import final, override

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from core.config import DiffusionConfig
from .base_vae import VAEBaseModel


@final
class Encoder(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, hidden3_dim: int, latent_dim: int
    ) -> None:
        super(Encoder, self).__init__()

        self.in_dim = in_dim

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 6,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim * 6),
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
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.conv(x)
        print(o.shape)
        o = self.flatten(o)
        print(o.shape)
        mu = self.fc_mu(o)
        logvar = self.fc_logvar(o)
        sigma = torch.exp(logvar * 0.5)

        return mu, sigma


@final
class Decoder(nn.Module):
    def __init__(
        self,
        latent_image_size: int,
        latent_dim: int,
        hidden3_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super(Decoder, self).__init__()

        self.latent_image_size = latent_image_size
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, hidden3_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim * 6,
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.fc(x)
        o = o.view(
            -1, self.hidden_dim * 6, self.latent_image_size, self.latent_image_size
        )
        o = self.deconv(o)

        return o


@final
class ConditionalVAE(VAEBaseModel):
    def __init__(self, config: DiffusionConfig, hidden_dim: int, latent_dim: int):
        super().__init__(config)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        if self.config.dataset == "mnist":
            self.image_size = 28
            self.channels = 1
        elif self.config.dataset == "cifar10" or self.config.dataset == "cifar100":
            self.image_size = 32
            self.channels = 3
        else:
            self.image_size = 224
            self.channels = 3
        self.latent_image_size = self.image_size // 8

        self.in_dim = self.channels
        self.hidden3_dim = (
            self.latent_image_size * self.latent_image_size * 6 * self.hidden_dim
        )

        self.encoder = Encoder(
            self.in_dim, self.hidden_dim, self.hidden3_dim, self.latent_dim
        )
        self.decoder = Decoder(
            self.latent_image_size,
            self.latent_dim,
            self.hidden3_dim,
            self.hidden_dim,
            self.in_dim,
        )

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
            x, _ = batch
            x = x.to(self.device)

            optimizer.zero_grad()

            mu, sigma = self.encoder(x)
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps

            x_hat = self.decoder(z)

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
                x, _ = batch
                x = x.to(self.device)

                mu, sigma = self.encoder(x)
                eps = torch.randn_like(sigma)
                z = mu + sigma * eps

                x_hat = self.decoder(z)

                loss = loss_function(x, x_hat, mu, sigma)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    @override
    def predict(self, batch_size: int = 64):
        self.logger.info(
            f"Training {self.config.model} on {self.config.dataset} dataset"
        )
        _ = self.eval()

        with torch.no_grad():
            z = torch.randn(
                (
                    batch_size,
                    self.latent_dim,
                )
            ).to(self.device)
            o = self.decoder(z)
            o = o.clamp(0, 1)
            o = o.cpu()

        grid_img = torchvision.utils.make_grid(o, nrow=8, padding=8, normalize=True)
        _ = plt.imshow(grid_img.permute(1, 2, 0))
        _ = plt.axis("off")
        _ = plt.show()

    @override
    def summary(self, input_size: tuple[int, int, int, int]):
        self.logger.info("Encoder summary")
        _ = summary(self.encoder, input_size)
        self.logger.info("Decoder summary")
        _ = summary(
            self.decoder,
            (
                input_size[0],
                self.latent_dim,
            ),
        )
