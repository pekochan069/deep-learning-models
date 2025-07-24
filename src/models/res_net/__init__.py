import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from core.dataset import get_num_classes
from models.base_model import BaseModel


# 3x3, 64
# 3x3. 64
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

    def forward(self, x: torch.Tensor):
        o = self.block1(x)
        o = self.block2(o)

        return self.relu(o + self.residual_conv(x))


# 1x1, 64
# 3x3, 64
# 1x1, 256
class ResBlockLarge(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, initial_block=False):
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

    def forward(self, x: torch.Tensor):
        o = self.block1(x)
        o = self.block2(o)
        o = self.block3(o)

        return self.relu(o + self.residual_conv(x))


class ResNet(BaseModel):
    def __init__(
        self,
        config: Config,
        res_blocks: list[ResBlock | ResBlockLarge],
        is_large_model=False,
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

    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)

        return o

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train the model."""
        self.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self(inputs)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        self.train(False)

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader: DataLoader, loss_function: nn.Module):
        """validate the model."""
        self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                loss = loss_function(outputs, targets)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    def predict(self, data_loader: DataLoader):
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                predictions = torch.argmax(outputs, dim=1)

                total += targets.size(0)
                correct += (predictions == targets).sum().item()

        accuracy = (correct / total) * 100
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")


class ResNet18(ResNet):
    def __init__(self, config: Config):
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


class ResNet34(ResNet):
    def __init__(self, config: Config):
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


class ResNet50(ResNet):
    def __init__(self, config: Config):
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


class ResNet101(ResNet):
    def __init__(self, config: Config):
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


class ResNet152(ResNet):
    def __init__(self, config: Config):
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
