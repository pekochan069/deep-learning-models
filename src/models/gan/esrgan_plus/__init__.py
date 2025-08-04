from typing import final, override

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms  # pyright: ignore[reportMissingTypeStubs]
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import GANConfig
from ..base_model import BaseGANModel


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
        self.lrelu = nn.LeakyReLU(0.2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv(x)
        o = self.bn(o)
        o = self.lrelu(o)

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

        # self.pool9 = nn.Linear(512 * image_width // 16 * image_height // 16, 1024)
        self.pool9 = nn.AdaptiveAvgPool2d(1)
        self.fc10 = nn.Linear(512, 1)

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

        o = self.pool9(o)
        o = o.view(o.size(0), -1)  # Flatten
        o = self.fc10(o)

        return o


@final
class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, is_last: bool):
        super(DenseLayer, self).__init__()
        self.is_last = is_last

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.lrelu = nn.LeakyReLU(0.2)

    @override
    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = [x]

        o = torch.cat(x, dim=1)

        o = self.conv(o)

        if not self.is_last:
            o = self.lrelu(o)

        return o


@final
class ResidualDenseBlock(nn.Module):
    def __init__(self):
        super(ResidualDenseBlock, self).__init__()
        for i in range(4):
            self.add_module(
                f"dense_layer_{i + 1}",
                DenseLayer(64 + 32 * i, False),
            )

        self.add_module("dense_layer_5", DenseLayer(192, True))

    @override
    def forward(self, x: torch.Tensor):
        features = [x]

        residual = x

        for i, layer in enumerate(self.children()):
            o = layer(features)
            features.append(o)

            if i == 1:
                o = o + residual
                residual = o

        o = torch.cat(features, dim=1)
        return o


@final
class RRDRB(nn.Module):
    def __init__(self, beta: float) -> None:
        super(RRDRB, self).__init__()

        self.dense_block1 = ResidualDenseBlock()
        self.transition_conv1 = nn.Conv2d(
            in_channels=224, out_channels=64, kernel_size=1
        )
        self.dense_block2 = ResidualDenseBlock()
        self.transition_conv2 = nn.Conv2d(
            in_channels=224, out_channels=64, kernel_size=1
        )
        self.dense_block3 = ResidualDenseBlock()
        self.transition_conv3 = nn.Conv2d(
            in_channels=224, out_channels=64, kernel_size=1
        )

        self.beta = beta

    @override
    def forward(self, x: torch.Tensor):
        o0 = x

        o = self.dense_block1(x)
        o = self.transition_conv1(o)
        o = o * self.beta
        o = o + o0

        o1 = o

        o = self.dense_block2(o)
        o = self.transition_conv2(o)
        o = o * self.beta
        o = o + o1

        o2 = o

        o = self.dense_block3(o)
        o = self.transition_conv3(o)
        o = o * self.beta
        o = o + o2

        o = o * self.beta

        o = o + o0

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
        self.lrelu = nn.LeakyReLU(0.2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv(x)
        o = self.pixel_shuffle(o)
        o = self.lrelu(o)

        return o


@final
class Generator(nn.Module):
    def __init__(self, rrdb_layers: int, beta: float):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False
            ),
            nn.LeakyReLU(0.2),
        )

        self.rrdb = nn.ModuleList([RRDRB(beta) for _ in range(rrdb_layers)])
        self.noise_sigma = nn.Parameter(torch.zeros(1, 64, 1, 1))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.upsample_block1 = UpsampleBlock()
        self.upsample_block2 = UpsampleBlock()

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, padding=4, bias=False
        )

    @override
    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        for layer in self.rrdb:
            o = layer(o)
            o += self.noise_sigma * torch.randn_like(o)
        o = self.layer2(o)
        o = self.upsample_block1(o)
        o = self.upsample_block2(o)

        o = self.conv3(o)

        return o


@final
class ESRGANPlus(BaseGANModel):
    def __init__(self, config: GANConfig):
        super(ESRGANPlus, self).__init__(config)

        beta: float = config.model_params["beta"] or 1.0
        assert 0 < beta <= 1

        rrdb_layers: int = config.model_params["rrdb_layers"] or 16
        assert 0 < rrdb_layers

        if config.dataset == "df2k_ost":
            image_width = 256
            image_height = 256
        elif config.dataset == "df2k_ost_small":
            image_width = 96
            image_height = 96
        else:
            image_width = 64
            image_height = 64

        self.discriminator = Discriminator(image_width, image_height)
        self.generator = Generator(rrdb_layers, beta)

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
        _ = self.generator.eval()

        with torch.no_grad():
            batch = next(iter(data_loader))
            z, targets = batch

            g_z = self.generator(z.to(self.device)).view(-1, 3, 96, 96)
            g_z = g_z.clamp(0, 1)  # Ensure pixel values are in [0, 1]
            g_z = g_z * 255.0  # Scale to [0, 255]
            g_z = g_z.byte()  # Convert to byte format

            generated_images = g_z.cpu()

        # show all generated_images and targets
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(transforms.ToPILImage()(targets[i].cpu()), aspect="auto")
            axes[0, i].axis("off")
            axes[1, i].imshow(
                transforms.ToPILImage()(generated_images[i]), aspect="auto"
            )
            axes[1, i].axis("off")
        axes[0, 0].set_title("Original")
        axes[1, 0].set_title("Generated")
        plt.tight_layout()
        plt.show()
