from typing import final, override
import torch
import torch.nn as nn

from core.config import ClassificationConfig
from ..base_model import ClassificationBaseModel


@final
class ExampleCNN(ClassificationBaseModel):
    def __init__(self, config: ClassificationConfig):
        super(ExampleCNN, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        fc1 = nn.Linear(128 * 4 * 4, 625)
        fc2 = nn.Linear(625, 10)
        _ = nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        self.layer4 = nn.Sequential(fc1, nn.ReLU(), nn.Dropout(0.5), fc2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = o.view(o.size(0), -1)
        o = self.layer4(o)
        return o
