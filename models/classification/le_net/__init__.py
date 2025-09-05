from typing import final, override
import torch
import torch.nn as nn

from core.config import ClassificationConfig
from core.registry import ModelRegistry
from ..base_model import ClassificationBaseModel


@ModelRegistry.register("le_net")
@final
class LeNet(ClassificationBaseModel):
    def __init__(self, config: ClassificationConfig):
        super(LeNet, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(16 * 5 * 5, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 10)
        _ = nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer3 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    @override
    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        o = self.layer2(o)
        o = o.view(o.size(0), -1)
        o = self.layer3(o)
        return o
