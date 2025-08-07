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
    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int):
        super(Encoder, self).__init__()

        self.in_channels = in_channels

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_channels, latent_channels, bias=False)
        self.fc_logvar = nn.Linear(hidden_channels, latent_channels, bias=False)

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.fc(x)

        mu = self.fc_mu(o)

        logvar = self.fc_logvar(o)
        sigma = torch.exp(logvar * 0.5)

        return mu, sigma


@final
class Decoder(nn.Module):
    def __init__(
        self, latent_channels: int, hidden_channels: int, out_channels: int
    ) -> None:
        super(Decoder, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels, bias=False),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=False)

    @override
    def forward(self, x: torch.Tensor):
        o = self.fc1(x)
        o = self.fc2(o)

        return o


@final
class VAE(DiffusionBaseModel):
    def __init__(self, config: DiffusionConfig):
        super(VAE, self).__init__(config)

        if self.config.dataset == "mnist":
            self.image_size = 28
        elif self.config.dataset == "cifar10" or self.config.dataset == "cifar100":
            self.image_size = 32
        elif (
            self.config.dataset == "imagenet" or self.config.dataset == "mini_imagenet"
        ):
            self.image_size = 224
        else:
            self.image_size = 32

        self.in_channels = self.image_size**2
        self.hidden_channels = 200
        self.latent_channels = 20

        self.encoder = Encoder(
            self.in_channels, self.hidden_channels, self.latent_channels
        )
        self.decoder = Decoder(
            self.latent_channels, self.hidden_channels, self.in_channels
        )

    @override
    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        x_hat = self.decoder(z)

        return x_hat

    @override
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train the model."""
        _ = self.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

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
        _ = self.train(False)

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
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

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
            z = torch.randn(batch_size, self.latent_channels, device=self.device)

            o: torch.Tensor = self.decoder(z)
            o = o.clamp(0, 1)
            o = o.view((batch_size, 1, self.image_size, self.image_size))

        grid_img = torchvision.utils.make_grid(o, nrow=8, padding=8, normalize=True)
        _ = plt.imshow(grid_img.permute(1, 2, 0))
        _ = plt.axis("off")
        _ = plt.show()

    @override
    def summary(self, input_size: tuple[int, int, int, int]):
        self.logger.info("Encoder summary")
        _ = summary(self.encoder, (self.in_channels,))
        self.logger.info("Decoder summary")
        _ = summary(self.decoder, (self.latent_channels,))
