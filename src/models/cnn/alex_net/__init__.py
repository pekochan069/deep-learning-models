from typing import final, override
import torch
import torch.nn as nn

from core.config import CNNConfig
from core.dataset import get_num_classes
from ..base_model import BaseCNNModel


@final
class AlexNet(BaseCNNModel):
    def __init__(self, config: CNNConfig):
        super(AlexNet, self).__init__(config)
        self.num_classes = get_num_classes(config.dataset)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        fc1 = nn.Linear(256 * 5 * 5, 4096)
        fc2 = nn.Linear(4096, 4096)
        fc3 = nn.Linear(4096, self.num_classes)
        _ = nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer4 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    @override
    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = o.view(o.size(0), -1)
        o = self.layer4(o)
        return o
