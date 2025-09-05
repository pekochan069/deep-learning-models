from typing import final, override
import torch
import torch.nn as nn

from core.config import ClassificationConfig
from core.dataset import get_num_classes
from core.registry import ModelRegistry
from ..base_model import ClassificationBaseModel


@ModelRegistry.register("vgg_net_11")
@final
class VGGNet11(ClassificationBaseModel):
    def __init__(self, config: ClassificationConfig):
        super(VGGNet11, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.LocalResponseNorm(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        self.num_classes = get_num_classes(config.dataset)
        fc3 = nn.Linear(4096, self.num_classes)
        _ = nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
        return o


@ModelRegistry.register("vgg_net_13")
@final
class VGGNet13(ClassificationBaseModel):
    def __init__(self, config: ClassificationConfig):
        super(VGGNet13, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        self.num_classes = get_num_classes(config.dataset)
        fc3 = nn.Linear(4096, self.num_classes)
        _ = nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
        return o


@ModelRegistry.register("vgg_net_16")
@final
class VGGNet16(ClassificationBaseModel):
    def __init__(self, config: ClassificationConfig):
        super(VGGNet16, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        self.num_classes = get_num_classes(config.dataset)
        fc3 = nn.Linear(4096, self.num_classes)
        _ = nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
        return o


@ModelRegistry.register("vgg_net_19")
@final
class VGGNet19(ClassificationBaseModel):
    def __init__(self, config: ClassificationConfig):
        super(VGGNet19, self).__init__(config)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        fc1 = nn.Linear(512 * 1 * 1, 4096)
        fc2 = nn.Linear(4096, 4096)
        self.num_classes = get_num_classes(config.dataset)
        fc3 = nn.Linear(4096, self.num_classes)
        _ = nn.init.kaiming_uniform_(fc1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(fc3.weight, nonlinearity="relu")
        self.layer6 = nn.Sequential(
            fc1, nn.ReLU(), nn.Dropout(0.5), fc2, nn.ReLU(), nn.Dropout(0.5), fc3
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        o = self.layer5(o)
        o = o.view(o.size(0), -1)
        o = self.layer6(o)
        return o
