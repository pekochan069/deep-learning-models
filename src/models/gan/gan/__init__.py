from typing import Any, final, override
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import GANConfig

from ..base_model import BaseGANModel


@final
class Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super(Discriminator, self).__init__()

        self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.linear1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 1)

        _ = nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity="relu")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.activation_fn(self.bn1(self.linear1(x)))
        o = self.dropout(o)
        o = self.activation_fn(self.bn2(self.linear2(o)))
        o = self.dropout(o)
        o = self.linear3(o)
        o = o.squeeze(dim=1)

        return o


@final
class Generator(nn.Module):
    def __init__(self, output_size: int = 784):
        super(Generator, self).__init__()

        self.activation_fn = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.linear1 = nn.Linear(100, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.linear4 = nn.Linear(1024, output_size)

        self.output_activation = nn.Tanh()  # Use Tanh for output layer

        _ = nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear4.weight, nonlinearity="relu")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.activation_fn(self.bn1(self.linear1(x)))
        o = self.dropout(o)
        o = self.activation_fn(self.bn2(self.linear2(o)))
        o = self.dropout(o)
        o = self.activation_fn(self.bn3(self.linear3(o)))
        o = self.linear4(o)
        o = self.output_activation(o)

        return o


@final
class GAN(BaseGANModel):
    def __init__(self, config: GANConfig):
        super(GAN, self).__init__(config)

        g_output_size = (
            784
            if config.dataset == "mnist"
            else 1024
            if config.dataset == "cifar10" or config.dataset == "cifar100"
            else 50176
        )

        self.generator = Generator(output_size=g_output_size)
        self.discriminator = Discriminator(g_output_size)

    @override
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        g_loss_function: nn.Module,
        d_loss_function: nn.Module,
    ) -> tuple[float, float]:
        """Train the model."""
        _ = self.generator.train()
        _ = self.discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Generator
            g_optimizer.zero_grad()
            z = torch.randn(inputs.size(0), 100, device=self.device)

            g_z = self.generator(z)
            d_g_z = self.discriminator(g_z)
            g_loss = g_loss_function(
                d_g_z,
                torch.full_like(d_g_z, self.config.real_label, device=self.device),
            )

            g_loss.backward()
            g_optimizer.step()

            # Discriminator
            d_optimizer.zero_grad()

            d_x = self.discriminator(inputs.reshape(inputs.size(0), -1))
            d_g_z = self.discriminator(g_z.detach())
            d_loss = d_loss_function(d_x, d_g_z)

            d_loss.backward()
            d_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        _ = self.generator.train(False)
        _ = self.discriminator.train(False)

        return epoch_g_loss / len(train_loader), epoch_d_loss / len(train_loader)

    @override
    def validate_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        g_loss_function: nn.Module,
        d_loss_function: nn.Module,
    ) -> tuple[float, float]:
        """validate the model."""
        _ = self.eval()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, _ = batch
                inputs = inputs.to(self.device)

                z = torch.randn(inputs.size(0), 100, device=self.device)
                g_z = self.generator(z)
                d_g_z = self.discriminator(g_z)
                g_loss = g_loss_function(
                    d_g_z,
                    torch.full_like(d_g_z, self.config.real_label, device=self.device),
                )

                d_x = self.discriminator(inputs.reshape(inputs.size(0), -1))
                d_g_z = self.discriminator(g_z)
                d_loss = d_loss_function(d_x, d_g_z)

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

        return epoch_g_loss / len(val_loader), epoch_d_loss / len(val_loader)

    @override
    def predict(
        self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Any:
        _ = self.discriminator.eval()
        _ = self.generator.eval()

        with torch.no_grad():
            z = torch.randn(128, 100, device=self.device)
            generated_images = self.generator(z)
            generated_images = generated_images.view(-1, 1, 28, 28)

        images = generated_images.cpu()
        # show images using matplotlib
        grid_size = int(images.size(0) ** 0.5)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < images.size(0):
                    axes[i, j].imshow(images[idx].permute(1, 2, 0).numpy(), cmap="gray")
                axes[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(f"images/{self.config.name}_generated_images.png")
        plt.show()
