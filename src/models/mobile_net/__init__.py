import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from core.dataset import get_num_classes

from ..base_model import BaseModel


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU6()

        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU6()

    def forward(self, x: torch.Tensor):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu2(o)

        return o


class MobileNet(BaseModel):
    def __init__(self, config: Config):
        super(MobileNet, self).__init__(config)
        self.num_classes = get_num_classes(config.dataset)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU6()

        self.dwconv2 = DepthwiseConv2d(in_channels=32, out_channels=64)
        self.dwconv3 = DepthwiseConv2d(in_channels=64, out_channels=128, stride=2)
        self.dwconv4 = DepthwiseConv2d(in_channels=128, out_channels=128)
        self.dwconv5 = DepthwiseConv2d(in_channels=128, out_channels=256, stride=2)
        self.dwconv6 = DepthwiseConv2d(in_channels=256, out_channels=256)
        self.dwconv7 = DepthwiseConv2d(in_channels=256, out_channels=512, stride=2)
        self.dwconv8 = DepthwiseConv2d(in_channels=512, out_channels=512)
        self.dwconv9 = DepthwiseConv2d(in_channels=512, out_channels=512)
        self.dwconv10 = DepthwiseConv2d(in_channels=512, out_channels=512)
        self.dwconv11 = DepthwiseConv2d(in_channels=512, out_channels=512)
        self.dwconv12 = DepthwiseConv2d(in_channels=512, out_channels=512)
        self.dwconv13 = DepthwiseConv2d(in_channels=512, out_channels=1024, stride=2)
        self.dwconv14 = DepthwiseConv2d(in_channels=1024, out_channels=1024, stride=1)

        self.pool15 = nn.AdaptiveAvgPool2d((1, 1))
        self.bn15 = nn.BatchNorm2d(1024)
        self.relu15 = nn.ReLU6()

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(1024, self.num_classes)
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)

        o = self.dwconv2(o)
        o = self.dwconv3(o)
        o = self.dwconv4(o)
        o = self.dwconv5(o)
        o = self.dwconv6(o)
        o = self.dwconv7(o)
        o = self.dwconv8(o)
        o = self.dwconv9(o)
        o = self.dwconv10(o)
        o = self.dwconv11(o)
        o = self.dwconv12(o)
        o = self.dwconv13(o)
        o = self.dwconv14(o)

        o = self.pool15(o)
        o = self.bn15(o)
        o = self.relu15(o)
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
