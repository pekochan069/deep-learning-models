from typing import final, override
import torch
import torch.nn as nn

from core.config import CNNConfig
from core.dataset import get_num_classes
from ..base_model import CNNBaseModel


# 3x3, 64
# 3x3. 64
@final
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResBlock, self).__init__()

        stride = 1 if in_channels == out_channels else 2

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.residual_conv = nn.Identity()

        self.relu = nn.ReLU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.block1(x)
        o = self.block2(o)

        return self.relu(o + self.residual_conv(x))


# 1x1, 64
# 3x3, 64
# 1x1, 256
@final
class ResBlockLarge(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, initial_block: bool = False
    ):
        super(ResBlockLarge, self).__init__()

        # stride = 1 if in_channels == out_channels * 4 and not initial_block else 2
        stride = 2 if in_channels != out_channels * 4 and not initial_block else 1

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1
            ),
            nn.BatchNorm2d(out_channels * 4),
        )

        if in_channels != out_channels * 4:
            self.residual_conv = nn.Conv2d(
                in_channels,
                out_channels * 4,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.residual_conv = nn.Identity()

        self.relu = nn.ReLU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.block1(x)
        o = self.block2(o)
        o = self.block3(o)

        return self.relu(o + self.residual_conv(x))


class ResNet(CNNBaseModel):
    num_classes: int
    layer1: nn.Sequential
    layer2: nn.Sequential
    layer3: nn.Sequential

    def __init__(
        self,
        config: CNNConfig,
        res_blocks: list[ResBlock | ResBlockLarge],
        is_large_model: bool = False,
    ):
        super(ResNet, self).__init__(config)
        self.num_classes = get_num_classes(self.config.dataset)

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(*res_blocks)

        final_channels = 2048 if is_large_model else 512

        self.layer3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_channels, self.num_classes),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)

        return o


@final
class ResNet18(ResNet):
    def __init__(self, config: CNNConfig):
        super(ResNet18, self).__init__(
            config,
            [
                ResBlock(64, 64),
                ResBlock(64, 64),
                ResBlock(64, 128),
                ResBlock(128, 128),
                ResBlock(128, 256),
                ResBlock(256, 256),
                ResBlock(256, 512),
                ResBlock(512, 512),
            ],
        )


@final
class ResNet34(ResNet):
    def __init__(self, config: CNNConfig):
        super(ResNet34, self).__init__(
            config,
            [
                ResBlock(64, 64),
                ResBlock(64, 64),
                ResBlock(64, 64),
                ResBlock(64, 128),
                ResBlock(128, 128),
                ResBlock(128, 128),
                ResBlock(128, 128),
                ResBlock(128, 256),
                ResBlock(256, 256),
                ResBlock(256, 256),
                ResBlock(256, 256),
                ResBlock(256, 256),
                ResBlock(256, 256),
                ResBlock(256, 512),
                ResBlock(512, 512),
                ResBlock(512, 512),
            ],
        )


@final
class ResNet50(ResNet):
    def __init__(self, config: CNNConfig):
        blocks = []

        # Stage 1: [3 blocks]
        blocks.extend(
            [
                ResBlockLarge(64, 64, initial_block=True),
                ResBlockLarge(256, 64),
                ResBlockLarge(256, 64),
            ]
        )

        # Stage 2: [4 blocks]
        blocks.extend(
            [
                ResBlockLarge(256, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
            ]
        )

        # Stage 3: [6 blocks]
        blocks.extend(
            [
                ResBlockLarge(512, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
            ]
        )

        # Stage 4: [3 blocks]
        blocks.extend(
            [
                ResBlockLarge(1024, 512),
                ResBlockLarge(2048, 512),
                ResBlockLarge(2048, 512),
            ]
        )

        super(ResNet50, self).__init__(config, blocks, is_large_model=True)


@final
class ResNet101(ResNet):
    def __init__(self, config: CNNConfig):
        blocks = []

        # Stage 1: [3 blocks]
        blocks.extend(
            [
                ResBlockLarge(64, 64, initial_block=True),
                ResBlockLarge(256, 64),
                ResBlockLarge(256, 64),
            ]
        )

        # Stage 2: [4 blocks]
        blocks.extend(
            [
                ResBlockLarge(256, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
            ]
        )

        # Stage 3: [23 blocks]
        blocks.extend(
            [
                ResBlockLarge(512, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
            ]
        )

        # Stage 4: [3 blocks]
        blocks.extend(
            [
                ResBlockLarge(1024, 512),
                ResBlockLarge(2048, 512),
                ResBlockLarge(2048, 512),
            ]
        )

        super(ResNet101, self).__init__(config, blocks, is_large_model=True)


@final
class ResNet152(ResNet):
    def __init__(self, config: CNNConfig):
        blocks = []

        # Stage 1: [3 blocks]
        blocks.extend(
            [
                ResBlockLarge(64, 64, initial_block=True),
                ResBlockLarge(256, 64),
                ResBlockLarge(256, 64),
            ]
        )

        # Stage 2: [8 blocks]
        blocks.extend(
            [
                ResBlockLarge(256, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
                ResBlockLarge(512, 128),
            ]
        )

        # Stage 3: [36 blocks]
        blocks.extend(
            [
                ResBlockLarge(512, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
                ResBlockLarge(1024, 256),
            ]
        )

        # Stage 4: [3 blocks]
        blocks.extend(
            [
                ResBlockLarge(1024, 512),
                ResBlockLarge(2048, 512),
                ResBlockLarge(2048, 512),
            ]
        )

        super(ResNet152, self).__init__(
            config,
            blocks,
            is_large_model=True,
        )
