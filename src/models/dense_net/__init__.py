import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from core.dataset import get_num_classes

from ..base_model import BaseModel


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super(DenseLayer, self).__init__()

        hidden_channels = growth_rate * bn_size

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=1
        )

        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=growth_rate,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: list[torch.Tensor] | torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = [x]

        o = torch.cat(x, dim=1)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.conv1(o)
        o = self.bn2(o)
        o = self.relu2(o)
        o = self.conv2(o)

        return o


class DenseBlock(nn.Module):
    def __init__(
        self, num_Layers: int, in_channels: int, growth_rate: int, bn_size: int = 4
    ):
        super(DenseBlock, self).__init__()

        for i in range(num_Layers):
            self.add_module(
                f"dense_layer_{i + 1}",
                DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size),
            )

    def forward(self, x: torch.Tensor):
        features = [x]

        for layer in self.children():
            o = layer(features)
            features.append(o)

        o = torch.cat(features, dim=1)
        return o


class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int):
        super(TransitionLayer, self).__init__()

        self.activation_fn = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            stride=1,
        )
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        o = self.bn1(x)
        o = self.activation_fn(o)
        o = self.conv1(o)
        o = self.bn2(o)
        o = self.activation_fn(o)
        o = self.pool2(o)

        return o


class DenseNet(BaseModel):
    def __init__(self, layers: list[nn.Module], out_channels: int, config: Config):
        super(DenseNet, self).__init__(config)

        self.num_classes: int = get_num_classes(config.dataset)
        self.growth_rate: int = self.config.model_params.get("growth_rate", 32)
        self.bn_size: int = self.config.model_params.get("bn_size", 4)
        self.compression: float = self.config.model_params.get("compression", 0.5)

        self.activation_fn = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.layers = nn.Sequential(*layers)

        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(out_channels, self.num_classes)
        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.activation_fn(o)

        o = self.pool2(o)
        o = self.bn2(o)
        o = self.activation_fn(o)

        for layer in self.layers:
            o = layer(o)

        o = self.pool4(o)
        o = self.bn4(o)
        o = self.activation_fn(o)

        o = o.view(o.size(0), -1)
        o = self.dropout(o)
        o = self.fc(o)

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


class DenseNetCifar(DenseNet):
    def __init__(self, config: Config):
        self.growth_rate = config.model_params.get("growth_rate", 32)
        out_channels = (
            ((64 + 6 * self.growth_rate) // 2 + 6 * self.growth_rate) // 2
        ) + 6 * self.growth_rate

        super(DenseNetCifar, self).__init__(
            [
                DenseBlock(
                    num_Layers=6,
                    in_channels=64,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(64 + 6 * self.growth_rate),
                DenseBlock(
                    num_Layers=6,
                    in_channels=(64 + 6 * self.growth_rate) // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    (64 + 6 * self.growth_rate) // 2 + 6 * self.growth_rate
                ),
                DenseBlock(
                    num_Layers=6,
                    in_channels=(
                        (64 + 6 * self.growth_rate) // 2 + 6 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
            ],
            out_channels,
            config,
        )


class DenseNet121(DenseNet):
    def __init__(self, config: Config):
        self.growth_rate = config.model_params.get("growth_rate", 32)
        out_channels = (
            ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
            + 24 * self.growth_rate
        ) // 2 + 16 * self.growth_rate

        super(DenseNet121, self).__init__(
            [
                DenseBlock(
                    num_Layers=6,
                    in_channels=64,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(64 + 6 * self.growth_rate),
                DenseBlock(
                    num_Layers=12,
                    in_channels=(64 + 6 * self.growth_rate) // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    (64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate
                ),
                DenseBlock(
                    num_Layers=24,
                    in_channels=(
                        (64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
                    + 24 * self.growth_rate,
                ),
                DenseBlock(
                    num_Layers=16,
                    in_channels=(
                        ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
                        + 24 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
            ],
            out_channels,
            config,
        )


class DenseNet169(DenseNet):
    def __init__(self, config: Config):
        self.growth_rate = config.model_params.get("growth_rate", 32)
        out_channels = (
            ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
            + 32 * self.growth_rate
        ) // 2 + 32 * self.growth_rate

        super(DenseNet169, self).__init__(
            [
                DenseBlock(
                    num_Layers=6,
                    in_channels=64,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(64 + 6 * self.growth_rate),
                DenseBlock(
                    num_Layers=12,
                    in_channels=(64 + 6 * self.growth_rate) // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    (64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate
                ),
                DenseBlock(
                    num_Layers=32,
                    in_channels=(
                        (64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
                    + 32 * self.growth_rate,
                ),
                DenseBlock(
                    num_Layers=32,
                    in_channels=(
                        ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
                        + 32 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
            ],
            out_channels,
            config,
        )


class DenseNet201(DenseNet):
    def __init__(self, config: Config):
        self.growth_rate = config.model_params.get("growth_rate", 32)
        out_channels = (
            ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
            + 48 * self.growth_rate
        ) // 2 + 32 * self.growth_rate

        super(DenseNet201, self).__init__(
            [
                DenseBlock(
                    num_Layers=6,
                    in_channels=64,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(64 + 6 * self.growth_rate),
                DenseBlock(
                    num_Layers=12,
                    in_channels=(64 + 6 * self.growth_rate) // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    (64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate
                ),
                DenseBlock(
                    num_Layers=48,
                    in_channels=(
                        (64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
                    + 48 * self.growth_rate,
                ),
                DenseBlock(
                    num_Layers=32,
                    in_channels=(
                        ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
                        + 48 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
            ],
            out_channels,
            config,
        )


class DenseNet264(DenseNet):
    def __init__(self, config: Config):
        self.growth_rate = config.model_params.get("growth_rate", 32)
        out_channels = (
            ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
            + 64 * self.growth_rate
        ) // 2 + 48 * self.growth_rate

        super(DenseNet264, self).__init__(
            [
                DenseBlock(
                    num_Layers=6,
                    in_channels=64,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(64 + 6 * self.growth_rate),
                DenseBlock(
                    num_Layers=12,
                    in_channels=(64 + 6 * self.growth_rate) // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    (64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate
                ),
                DenseBlock(
                    num_Layers=64,
                    in_channels=(
                        (64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
                TransitionLayer(
                    ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
                    + 64 * self.growth_rate,
                ),
                DenseBlock(
                    num_Layers=48,
                    in_channels=(
                        ((64 + 6 * self.growth_rate) // 2 + 12 * self.growth_rate) // 2
                        + 64 * self.growth_rate
                    )
                    // 2,
                    growth_rate=self.growth_rate,
                ),
            ],
            out_channels,
            config,
        )
