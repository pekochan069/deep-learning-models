from typing import final, override
import torch
import torch.nn as nn


from core.config import CNNConfig
from core.dataset import get_num_classes
from ..base_model import CNNBaseModel


@final
class InceptionV1Module(nn.Module):
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
        super(InceptionV1Module, self).__init__()

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

    @override
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


@final
class InceptionV1(CNNBaseModel):
    def __init__(self, config: CNNConfig):
        super(InceptionV1, self).__init__(config)
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

        self.module1 = InceptionV1Module(192, 64, 96, 128, 16, 32, 32)
        self.module2 = InceptionV1Module(256, 128, 128, 192, 32, 96, 64)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.module3 = InceptionV1Module(480, 192, 96, 208, 16, 48, 64)
        self.module4 = InceptionV1Module(512, 160, 112, 224, 24, 64, 64)
        self.module5 = InceptionV1Module(512, 128, 128, 256, 24, 64, 64)
        self.module6 = InceptionV1Module(512, 112, 144, 288, 32, 64, 64)
        self.module7 = InceptionV1Module(528, 256, 160, 320, 32, 128, 128)

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.module8 = InceptionV1Module(832, 256, 160, 320, 32, 128, 128)
        self.module9 = InceptionV1Module(832, 384, 192, 384, 48, 128, 128)

        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, self.num_classes)

        _ = torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

    @override
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
