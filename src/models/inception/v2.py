import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config
from core.dataset import get_num_classes

from ..base_model import BaseModel


class InceptionV2Module1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        reduce_3x3: int,
        out_3x3: int,
        reduce_3x3_2: int,
        out_3x3_2: int,
        pool_proj: int,
    ):
        super(InceptionV2Module1, self).__init__()

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
            in_channels=in_channels, out_channels=reduce_3x3_2, kernel_size=1
        )
        self.bn3_1 = nn.BatchNorm2d(reduce_3x3_2)
        self.conv3_2 = nn.Conv2d(
            in_channels=reduce_3x3_2,
            out_channels=out_3x3_2,
            kernel_size=5,
            padding=2,
        )
        self.bn3_2 = nn.BatchNorm2d(out_3x3_2)
        self.conv3_3 = nn.Conv2d(
            in_channels=out_3x3_2, out_channels=out_3x3_2, kernel_size=3, padding=1
        )
        self.bn3_3 = nn.BatchNorm2d(out_3x3_2)

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
        o3 = self.conv3_3(o3)
        o3 = self.bn3_3(o3)
        o3 = self.activation_fn(o3)

        o4 = self.pool4_1(x)
        o4 = self.bn4_1(o4)
        o4 = self.activation_fn(o4)
        o4 = self.conv4_2(o4)
        o4 = self.bn4_2(o4)
        o4 = self.activation_fn(o4)

        return torch.cat((o1, o2, o3, o4), dim=1)


class InceptionV2Module2(nn.Module):
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
        super(InceptionV2Module2, self).__init__()

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
            kernel_size=(1, 7),
            padding=(0, 3),
        )
        self.bn2_2 = nn.BatchNorm2d(out_3x3)
        self.conv2_3 = nn.Conv2d(
            in_channels=out_3x3,
            out_channels=out_3x3,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.bn2_3 = nn.BatchNorm2d(out_3x3)

        self.conv3_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=reduce_5x5, kernel_size=1
        )
        self.bn3_1 = nn.BatchNorm2d(reduce_5x5)
        self.conv3_2 = nn.Conv2d(
            in_channels=reduce_5x5,
            out_channels=out_5x5,
            kernel_size=(1, 7),
            padding=(0, 3),
        )
        self.bn3_2 = nn.BatchNorm2d(out_5x5)
        self.conv3_3 = nn.Conv2d(
            in_channels=out_5x5,
            out_channels=out_5x5,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.bn3_3 = nn.BatchNorm2d(out_5x5)
        self.conv3_4 = nn.Conv2d(
            in_channels=out_5x5,
            out_channels=out_5x5,
            kernel_size=(1, 7),
            padding=(0, 3),
        )
        self.bn3_4 = nn.BatchNorm2d(out_5x5)
        self.conv3_5 = nn.Conv2d(
            in_channels=out_5x5,
            out_channels=out_5x5,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.bn3_5 = nn.BatchNorm2d(out_5x5)

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
        o2 = self.conv2_3(o2)
        o2 = self.bn2_3(o2)
        o2 = self.activation_fn(o2)

        o3 = self.conv3_1(x)
        o3 = self.bn3_1(o3)
        o3 = self.activation_fn(o3)
        o3 = self.conv3_2(o3)
        o3 = self.bn3_2(o3)
        o3 = self.activation_fn(o3)
        o3 = self.conv3_3(o3)
        o3 = self.bn3_3(o3)
        o3 = self.activation_fn(o3)
        o3 = self.conv3_4(o3)
        o3 = self.bn3_4(o3)
        o3 = self.activation_fn(o3)
        o3 = self.conv3_5(o3)
        o3 = self.bn3_5(o3)
        o3 = self.activation_fn(o3)

        o4 = self.pool4_1(x)
        o4 = self.bn4_1(o4)
        o4 = self.activation_fn(o4)
        o4 = self.conv4_2(o4)
        o4 = self.bn4_2(o4)
        o4 = self.activation_fn(o4)

        return torch.cat((o1, o2, o3, o4), dim=1)


class InceptionV2Module3(nn.Module):
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
        super(InceptionV2Module3, self).__init__()

        self.activation_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_1x1, kernel_size=1
        )
        self.bn1 = nn.BatchNorm2d(out_1x1)

        self.conv2_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=reduce_3x3, kernel_size=1
        )
        self.bn2_1 = nn.BatchNorm2d(reduce_3x3)
        self.conv2_2_1 = nn.Conv2d(
            in_channels=reduce_3x3,
            out_channels=out_3x3,
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self.bn2_2_1 = nn.BatchNorm2d(out_3x3)
        self.conv2_2_2 = nn.Conv2d(
            in_channels=reduce_3x3,
            out_channels=out_3x3,
            kernel_size=(3, 1),
            padding=(1, 0),
        )
        self.bn2_2_2 = nn.BatchNorm2d(out_3x3)

        self.conv3_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=reduce_5x5, kernel_size=1
        )
        self.bn3_1 = nn.BatchNorm2d(reduce_5x5)
        self.conv3_2 = nn.Conv2d(
            in_channels=reduce_5x5,
            out_channels=out_5x5,
            kernel_size=3,
            padding=1,
        )
        self.bn3_2 = nn.BatchNorm2d(out_5x5)
        self.conv3_3_1 = nn.Conv2d(
            in_channels=out_5x5,
            out_channels=out_5x5,
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self.bn3_3_1 = nn.BatchNorm2d(out_5x5)
        self.conv3_3_2 = nn.Conv2d(
            in_channels=out_5x5,
            out_channels=out_5x5,
            kernel_size=(3, 1),
            padding=(1, 0),
        )
        self.bn3_3_2 = nn.BatchNorm2d(out_5x5)

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
        o2_1 = self.conv2_2_1(o2)
        o2_1 = self.bn2_2_1(o2_1)
        o2_1 = self.activation_fn(o2_1)
        o2_2 = self.conv2_2_2(o2)
        o2_2 = self.bn2_2_2(o2_2)
        o2_2 = self.activation_fn(o2_2)
        o2 = o2_1 + o2_2

        o3 = self.conv3_1(x)
        o3 = self.bn3_1(o3)
        o3 = self.activation_fn(o3)
        o3 = self.conv3_2(o3)
        o3 = self.bn3_2(o3)
        o3_1 = self.conv3_3_1(o3)
        o3_1 = self.bn3_3_1(o3_1)
        o3_1 = self.activation_fn(o3_1)
        o3_2 = self.conv3_3_2(o3)
        o3_2 = self.bn3_3_2(o3_2)
        o3_2 = self.activation_fn(o3_2)
        o3 = o3_1 + o3_2

        o4 = self.pool4_1(x)
        o4 = self.bn4_1(o4)
        o4 = self.activation_fn(o4)
        o4 = self.conv4_2(o4)
        o4 = self.bn4_2(o4)
        o4 = self.activation_fn(o4)

        return torch.cat((o1, o2, o3, o4), dim=1)


class InceptionV2(BaseModel):
    def __init__(self, config: Config):
        super(InceptionV2, self).__init__(config)
        self.num_classes = get_num_classes(config.dataset)

        self.activation_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(80)

        self.conv6 = nn.Conv2d(
            in_channels=80, out_channels=192, kernel_size=3, stride=2
        )
        self.bn6 = nn.BatchNorm2d(192)

        self.module7 = InceptionV2Module1(
            in_channels=192,
            out_1x1=64,
            reduce_3x3=64,
            out_3x3=96,
            reduce_3x3_2=64,
            out_3x3_2=64,
            pool_proj=32,
        )
        self.module8 = InceptionV2Module1(
            in_channels=256,
            out_1x1=64,
            reduce_3x3=64,
            out_3x3=96,
            reduce_3x3_2=64,
            out_3x3_2=96,
            pool_proj=64,
        )
        self.module9 = InceptionV2Module1(
            in_channels=320,
            out_1x1=64,
            reduce_3x3=96,
            out_3x3=96,
            reduce_3x3_2=96,
            out_3x3_2=128,
            pool_proj=96,
        )

        self.module10 = InceptionV2Module2(
            in_channels=384,
            out_1x1=128,
            reduce_3x3=128,
            out_3x3=160,
            reduce_5x5=32,
            out_5x5=64,
            pool_proj=96,
        )

        self.module11 = InceptionV2Module2(
            in_channels=448,
            out_1x1=224,
            reduce_3x3=64,
            out_3x3=96,
            reduce_5x5=96,
            out_5x5=128,
            pool_proj=128,
        )

        self.module12 = InceptionV2Module2(
            in_channels=576,
            out_1x1=192,
            reduce_3x3=96,
            out_3x3=128,
            reduce_5x5=96,
            out_5x5=128,
            pool_proj=128,
        )

        self.module13 = InceptionV2Module2(
            in_channels=576,
            out_1x1=160,
            reduce_3x3=128,
            out_3x3=160,
            reduce_5x5=128,
            out_5x5=160,
            pool_proj=96,
        )

        self.module14 = InceptionV2Module2(
            in_channels=576,
            out_1x1=96,
            reduce_3x3=128,
            out_3x3=192,
            reduce_5x5=160,
            out_5x5=192,
            pool_proj=96,
        )

        self.module15 = InceptionV2Module3(
            in_channels=576,
            out_1x1=128,
            reduce_3x3=128,
            out_3x3=192,
            reduce_5x5=192,
            out_5x5=192,
            pool_proj=256,
        )

        self.module16 = InceptionV2Module3(
            in_channels=768,
            out_1x1=352,
            reduce_3x3=192,
            out_3x3=320,
            reduce_5x5=192,
            out_5x5=224,
            pool_proj=128,
        )

        self.module17 = InceptionV2Module3(
            in_channels=1024,
            out_1x1=352,
            reduce_3x3=192,
            out_3x3=320,
            reduce_5x5=192,
            out_5x5=224,
            pool_proj=128,
        )

        self.pool18 = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, self.num_classes)

        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor):
        # i.shape = (3x224x224)
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.activation_fn(o)

        # i.shape = (32x112x112)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.activation_fn(o)

        # i.shape = (32x112x112)
        o = self.conv3(o)
        o = self.bn3(o)
        o = self.activation_fn(o)

        # i.shape = (64x56x56)
        o = self.pool4(o)
        o = self.bn4(o)
        o = self.activation_fn(o)

        # i.shape = (64x56x56)
        o = self.conv5(o)
        o = self.bn5(o)
        o = self.activation_fn(o)

        # i.shape = (80x56x56)
        o = self.conv6(o)
        o = self.bn6(o)
        o = self.activation_fn(o)

        # i.shape = (192x28x28)
        o = self.module7(o)
        o = self.module8(o)
        o = self.module9(o)
        o = self.module10(o)
        o = self.module11(o)
        o = self.module12(o)
        o = self.module13(o)
        o = self.module14(o)
        o = self.module15(o)
        o = self.module16(o)
        o = self.module17(o)

        o = self.pool18(o)

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
