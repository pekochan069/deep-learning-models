import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from models.base_model import BaseModel


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_shortcut=True):
        super(ResBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=use_shortcut and 1 or 2,
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
            nn.ReLU(),
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor):
        y = self.block1(x)
        y = self.block2(y)

        return y + self.residual_conv(x)


class ResNet(BaseModel):
    def __init__(self, config: Config, res_blocks: list[ResBlock]):
        super(ResNet, self).__init__(config, input_size=(3, 224, 224))
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(*res_blocks)
        self.layer3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=)
        )


class ResNet18(BaseModel):
    def __init__(self, config: Config):
        super(ResNet18, self).__init__(config)

        self.res_block1 = ResBlock(64, 64)
        self.res_block2 = ResBlock(64, 64)

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        fc3 = nn.Linear(4096, 100)
        nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
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
