from typing import final, override

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
    def __init__(self):
        super(Discriminator, self).__init__()

    @override
    def forward(self, x: torch.Tensor):
        o = x

        return o


@final
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    @override
    def forward(self, x: torch.Tensor):
        o = x

        return o


@final
class ESRGAN(BaseGANModel):
    def __init__(self, config: GANConfig):
        super(ESRGAN, self).__init__(config)

        self.discriminator = Discriminator()
        self.generator = Generator()

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
            z, targets = batch
            z = z.to(self.device)
            targets = targets.to(self.device)

            #######################
            # Train Discriminator #
            #######################

            d_optimizer.zero_grad()

            d_x = self.discriminator(targets)
            g_z = self.generator(z)
            d_g_z = self.discriminator(g_z.detach())

            d_loss = d_loss_function(d_x, d_g_z)

            d_loss.backward()
            d_optimizer.step()

            #######################
            #   Train Generator   #
            #######################

            g_optimizer.zero_grad()

            d_g_z = self.discriminator(g_z)

            g_loss = g_loss_function(targets, g_z, d_g_z)

            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        _ = self.generator.train(False)
        _ = self.discriminator.train(False)

        return epoch_g_loss / len(train_loader), epoch_d_loss / len(train_loader)

    @override
    def predict(self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]):
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
