import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from core.dataset import get_num_classes

from ..base_model import BaseModel


class InceptionV3Module(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        reduce_3x3: int,
        out_3x3: int,
        reduce_5x5: int,
        out_5x5: int,
        pool_proj: int,
    ):
        super(InceptionV3Module, self).__init__()

        self.activation_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_1x1, kernel_size=1
        )
        self.bn1 = nn.BatchNorm2d(out_1x1)

        self.conv2_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=reduce_3x3, kernel_size=1
        )
        self.bn2_1 = nn.BatchNorm2d(reduce_3x3)
        self.conv2_2 = nn.Conv2d(
            in_channels=reduce_3x3,
            out_channels=out_3x3,
            kernel_size=3,
            padding=1,
        )
        self.bn2_2 = nn.BatchNorm2d(out_3x3)

        self.conv3_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=reduce_5x5, kernel_size=1
        )
        self.bn3_1 = nn.BatchNorm2d(reduce_5x5)
        self.conv3_2 = nn.Conv2d(
            in_channels=reduce_5x5,
            out_channels=out_5x5,
            kernel_size=5,
            padding=2,
        )
        self.bn3_2 = nn.BatchNorm2d(out_5x5)

        self.pool4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(in_channels)
        self.conv4_2 = nn.Conv2d(
            in_channels=in_channels, out_channels=pool_proj, kernel_size=1
        )
        self.bn4_2 = nn.BatchNorm2d(pool_proj)

    def forward(self, x: torch.Tensor):
        o1 = self.conv1(x)
        o1 = self.bn1(o1)
        o1 = self.activation_fn(o1)

        o2 = self.conv2_1(x)
        o2 = self.bn2_1(o2)
        o2 = self.activation_fn(o2)
        o2 = self.conv2_2(o2)
        o2 = self.bn2_2(o2)
        o2 = self.activation_fn(o2)

        o3 = self.conv3_1(x)
        o3 = self.bn3_1(o3)
        o3 = self.activation_fn(o3)
        o3 = self.conv3_2(o3)
        o3 = self.bn3_2(o3)
        o3 = self.activation_fn(o3)

        o4 = self.pool4_1(x)
        o4 = self.bn4_1(o4)
        o4 = self.activation_fn(o4)
        o4 = self.conv4_2(o4)
        o4 = self.bn4_2(o4)
        o4 = self.activation_fn(o4)

        return torch.cat((o1, o2, o3, o4), dim=1)


class InceptionV3(BaseModel):
    def __init__(self, config: Config):
        super(InceptionV3, self).__init__(config)
        self.num_classes = get_num_classes(config.dataset)

        self.activation_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.module1 = InceptionV3Module(192, 64, 96, 128, 16, 32, 32)
        self.module2 = InceptionV3Module(256, 128, 128, 192, 32, 96, 64)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.module3 = InceptionV3Module(480, 192, 96, 208, 16, 48, 64)
        self.module4 = InceptionV3Module(512, 160, 112, 224, 24, 64, 64)
        self.module5 = InceptionV3Module(512, 128, 128, 256, 24, 64, 64)
        self.module6 = InceptionV3Module(512, 112, 144, 288, 32, 64, 64)
        self.module7 = InceptionV3Module(528, 256, 160, 320, 32, 128, 128)

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.module8 = InceptionV3Module(832, 256, 160, 320, 32, 128, 128)
        self.module9 = InceptionV3Module(832, 384, 192, 384, 48, 128, 128)

        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, self.num_classes)

        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.activation_fn(o)
        o = self.pool1(o)

        o = self.conv2(o)
        o = self.bn2(o)
        o = self.activation_fn(o)
        o = self.pool2(o)

        o = self.module1(o)
        o = self.module2(o)

        o = self.pool3(o)

        o = self.module3(o)
        o = self.module4(o)
        o = self.module5(o)
        o = self.module6(o)
        o = self.module7(o)

        o = self.pool4(o)

        o = self.module8(o)
        o = self.module9(o)

        o = self.pool5(o)

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
