from typing import final, override

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_model import BaseGANModel
from core.config import GANConfig


@final
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(DiscriminatorBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv(x)
        o = self.bn(o)
        o = self.prelu(o)

        return o


@final
class Discriminator(nn.Module):
    def __init__(self, image_width: int, image_height: int):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.block2 = DiscriminatorBlock(in_channels=64, out_channels=64, stride=2)
        self.block3 = DiscriminatorBlock(in_channels=64, out_channels=128, stride=1)
        self.block4 = DiscriminatorBlock(in_channels=128, out_channels=128, stride=2)
        self.block5 = DiscriminatorBlock(in_channels=128, out_channels=256, stride=1)
        self.block6 = DiscriminatorBlock(in_channels=256, out_channels=256, stride=2)
        self.block7 = DiscriminatorBlock(in_channels=256, out_channels=512, stride=1)
        self.block8 = DiscriminatorBlock(in_channels=512, out_channels=512, stride=2)

        self.fc9 = nn.Linear(512 * image_width // 16 * image_height // 16, 1024)
        self.lrelu9 = nn.LeakyReLU(0.2, inplace=True)
        self.fc10 = nn.Linear(1024, 1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv1(x)
        o = self.lrelu1(o)

        o = self.block2(o)
        o = self.block3(o)
        o = self.block4(o)
        o = self.block5(o)
        o = self.block6(o)
        o = self.block7(o)
        o = self.block8(o)

        o = o.view(o.size(0), -1)  # Flatten
        o = self.fc9(o)
        o = self.lrelu9(o)
        o = self.fc10(o)

        return o


@final
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.prelu1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = o + x  # Skip connection

        return o


@final
class UpsampleBlock(nn.Module):
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv(x)
        o = self.pixel_shuffle(o)
        o = self.prelu(o)

        return o


@final
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False
        )
        self.prelu1 = nn.PReLU()

        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(16)])

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.upsample_block1 = UpsampleBlock()
        self.upsample_block2 = UpsampleBlock()

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, padding=4, bias=False
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv1(x)
        o = self.prelu1(o)

        i = o

        for block in self.blocks:
            o = block(o)

        o = self.conv2(o)
        o = self.bn2(o)
        o = o + i

        o = self.upsample_block1(o)
        o = self.upsample_block2(o)

        o = self.conv3(o)

        return o


@final
class SRGAN(BaseGANModel):
    def __init__(self, config: GANConfig):
        super(SRGAN, self).__init__(config)

        image_width = 256
        image_height = 256

        self.generator = Generator()
        self.discriminator = Discriminator(
            image_width=image_width, image_height=image_height
        )

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
