from typing import final, override
import torch
import torch.nn as nn

from core.config import CNNConfig
from core.dataset import get_num_classes

from ..base_model import BaseCNNModel


@final
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu2(o)

        return o


@final
class ShuffleNetV1(BaseCNNModel):
    def __init__(self, config: CNNConfig):
        super(ShuffleNetV1, self).__init__(config)
        self.num_classes = get_num_classes(config.dataset)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

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
        self.dwconv14 = DepthwiseConv2d(in_channels=1024, out_channels=1024, stride=2)

        self.pool15 = nn.AdaptiveAvgPool2d((1, 1))
        self.bn15 = nn.BatchNorm2d(1024)
        self.relu15 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(1024, self.num_classes)
        _ = nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        o = self.fc(o)

        return o
