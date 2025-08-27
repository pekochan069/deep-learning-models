from typing import Any, final, override

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import DiffusionConfig
from models.diffusion.base_model import DiffusionBaseModel


@final
class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super(Encoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=False)

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.fc(x)

        mu = self.fc_mu(o)

        logvar = self.fc_logvar(o)
        sigma = torch.exp(logvar * 0.5)

        return mu, sigma


@final
class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int) -> None:
        super(Decoder, self).__init__()

        self.m = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    @override
    def forward(self, x: torch.Tensor):
        o = self.m(x)

        return o


@final
class SimpleVAE(DiffusionBaseModel):
    def __init__(self, config: DiffusionConfig):
        super(SimpleVAE, self).__init__(config)

        if self.config.dataset == "mnist":
            self.image_size = 28
        elif self.config.dataset == "cifar10" or self.config.dataset == "cifar100":
            self.image_size = 32
        else:
            self.image_size = 224

        self.in_dim = self.image_size**2
        self.hidden_dim = int(self.in_dim / 4)
        self.latent_dim = int(self.hidden_dim / 8)

        self.encoder = Encoder(self.in_dim, self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.in_dim)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.decoder(x)

        return o

    @override
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, _ = batch
            inputs = inputs.to(self.device)

            x = inputs.view((-1, self.in_channels))

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
    ) -> float:
        """Evaluate the model."""
        _ = self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, _ = batch
                inputs = inputs.to(self.device)

                x = inputs.view((-1, self.in_channels))

                mu, sigma = self.encoder(x)
                eps = torch.randn_like(sigma)
                z = mu + sigma * eps

                x_hat = self.decoder(z)
                loss = loss_function(x, x_hat, mu, sigma)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    @override
    def predict(
        self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Any:
        """Evaluate the model on the provided data loader."""
        _ = self.eval()

        batch_size = 64

        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim).to(self.device)

            o = self.decoder(z)
            o = o.clamp(0, 1)
            o = o.view((batch_size, 1, self.image_size, self.image_size))

        grid_img = torchvision.utils.make_grid(o, nrow=8, padding=8, normalize=True)
        _ = plt.imshow(grid_img.permute(1, 2, 0))
        _ = plt.axis("off")
        _ = plt.show()

    @override
    def summary(self, input_size: tuple[int, int, int, int]):
        self.logger.info("Encoder summary")
        _ = summary(self.encoder, (self.in_dim,))
        self.logger.info("Decoder summary")
        _ = summary(self.decoder, (self.latent_dim,))
