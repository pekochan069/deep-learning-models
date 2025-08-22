import math
from typing import final, override
import torch
import torch.nn as nn
from core.config import ClassificationConfig
from core.dataset import get_num_classes
from ..base_model import ClassificationBaseModel

alpha = 1.2
beta = 1.1
gamma = 1.15


def scale_params(phi: float):
    return alpha**phi, beta**phi, gamma**phi


@final
class MBConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        squeeze_and_excite: bool = True,
    ):
        super(MBConv2d, self).__init__()

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 1. Expansion - 1x1 Pointwise
        if expand_ratio == 1:
            expanded_channels = in_channels
            self.expansion = nn.Identity()
        else:
            expanded_channels = in_channels * expand_ratio
            self.expansion = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(),
            )

        # 2. Depthwise - 3x3 or 5x5 Depthwise
        self.conv2 = nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expanded_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.silu2 = nn.SiLU()

        # 3. Squeeze and Excitation
        if squeeze_and_excite:
            se_channels = max(1, expanded_channels // 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(se_channels, expanded_channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.se = nn.Identity()

        # 4. Projection - 1x1 Pointwise
        self.conv3 = nn.Conv2d(
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.expansion(x)

        o = self.conv2(o)
        o = self.bn2(o)
        o = self.silu2(o)

        o = self.se(o) * o

        o = self.conv3(o)
        o = self.bn3(o)

        if self.use_residual:
            o += x

        return o


def make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """채널 수를 8의 배수로 만들어 하드웨어 효율성 향상"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_layers(width_multiplier: float, depth_multiplier: float):
    stage_configs = [
        # [in_channels, out_channels, kernel_size, stride, expand_ratio, num_layers]
        [32, 16, 3, 1, 1, 1],  # Stage 2
        [16, 24, 3, 1, 6, 2],  # Stage 3
        [24, 40, 5, 2, 6, 2],  # Stage 4
        [40, 80, 3, 2, 6, 3],  # Stage 5
        [80, 112, 5, 2, 6, 3],  # Stage 6
        [112, 192, 5, 1, 6, 4],  # Stage 7
        [192, 320, 3, 2, 6, 1],  # Stage 8
    ]

    layers = nn.ModuleList()

    for stage_config in stage_configs:
        in_channels, out_channels, kernel_size, stride, expand_ratio, num_layers = (
            stage_config
        )

        # 채널 수에 width multiplier 적용
        in_channels = make_divisible(in_channels * width_multiplier)
        out_channels = make_divisible(out_channels * width_multiplier)

        # layer 수에 depth multiplier 적용
        num_layers = max(1, int(math.ceil(num_layers * depth_multiplier)))

        # 첫 번째 layer (stride 적용)
        _ = layers.append(
            MBConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                expand_ratio=expand_ratio,
            )
        )

        # 나머지 layers (stride=1)
        for _ in range(num_layers - 1):
            _ = layers.append(
                MBConv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    expand_ratio=expand_ratio,
                )
            )

    final_channels = make_divisible(320 * width_multiplier)
    return layers, final_channels


class EfficientNetV1(ClassificationBaseModel):
    width_multiplier: float
    depth_multiplier: float
    num_classes: int
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    silu1: nn.SiLU
    layers: nn.ModuleList
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d
    silu2: nn.SiLU
    pool3: nn.AdaptiveAvgPool2d
    dropout: nn.Dropout
    fc: nn.Linear

    def __init__(
        self,
        config: ClassificationConfig,
        # layers: nn.ModuleList,
        # layers_out_channels: int,
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        dropdown: float = 0.2,
    ):
        super(EfficientNetV1, self).__init__(config)
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.num_classes = get_num_classes(config.dataset)

        layers, layers_out_channels = get_layers(width_multiplier, depth_multiplier)

        if config.dataset == "cifar10" or config.dataset == "cifar100":
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        self.bn1 = nn.BatchNorm2d(32)
        self.silu1 = nn.SiLU()

        self.layers = layers

        self.conv2 = nn.Conv2d(
            in_channels=layers_out_channels,
            out_channels=1280,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.silu2 = nn.SiLU()

        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(dropdown)
        self.fc = nn.Linear(1280, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _ = nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    _ = nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                _ = nn.init.ones_(m.weight)
                _ = nn.init.zeros_(m.bias)
                m.momentum = 0.01
                m.eps = 1e-3
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                _ = nn.init.uniform_(m.weight, -init_range, init_range)
                _ = nn.init.zeros_(m.bias)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.silu1(o)

        for layer in self.layers:
            o = layer(o)

        o = self.conv2(o)
        o = self.bn2(o)
        o = self.silu2(o)
        o = self.pool3(o)
        o = o.view(o.size(0), -1)  # Flatten the tensor
        o = self.dropout(o)
        o = self.fc(o)

        return o


@final
class EfficientNetV1B0(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B0, self).__init__(
            config, width_multiplier=1.0, depth_multiplier=1.0, dropdown=0.2
        )


@final
class EfficientNetV1B1(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B1, self).__init__(
            config, width_multiplier=1.0, depth_multiplier=1.1, dropdown=0.2
        )


@final
class EfficientNetV1B2(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B2, self).__init__(
            config, width_multiplier=1.1, depth_multiplier=1.2, dropdown=0.3
        )


@final
class EfficientNetV1B3(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B3, self).__init__(
            config, width_multiplier=1.2, depth_multiplier=1.4, dropdown=0.3
        )


@final
class EfficientNetV1B4(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B4, self).__init__(
            config, width_multiplier=1.4, depth_multiplier=1.8, dropdown=0.4
        )


@final
class EfficientNetV1B5(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B5, self).__init__(
            config, width_multiplier=1.6, depth_multiplier=2.2, dropdown=0.4
        )


@final
class EfficientNetV1B6(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B6, self).__init__(
            config, width_multiplier=1.8, depth_multiplier=2.6, dropdown=0.5
        )


@final
class EfficientNetV1B7(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B7, self).__init__(
            config, width_multiplier=2.0, depth_multiplier=3.1, dropdown=0.5
        )


@final
class EfficientNetV1B8(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1B8, self).__init__(
            config, width_multiplier=2.2, depth_multiplier=3.6, dropdown=0.5
        )


@final
class EfficientNetV1L2(EfficientNetV1):
    def __init__(self, config: ClassificationConfig):
        super(EfficientNetV1L2, self).__init__(
            config, width_multiplier=4.3, depth_multiplier=5.3, dropdown=0.5
        )


# class EfficientNetV1B0(EfficientNetV1):
#     def __init__(self, config: ClassificationConfig):
#         layers = nn.ModuleList(
#             [
#                 # 1
#                 MBConv2d(32, 16, kernel_size=3, stride=2, expand_ratio=1),
#                 # 2
#                 MBConv2d(16, 24, kernel_size=3, stride=1, expand_ratio=6),
#                 MBConv2d(24, 24, kernel_size=3, stride=1, expand_ratio=6),
#                 # 3
#                 MBConv2d(24, 40, kernel_size=5, stride=2, expand_ratio=6),
#                 MBConv2d(40, 40, kernel_size=5, stride=1, expand_ratio=6),
#                 # 4
#                 MBConv2d(40, 80, kernel_size=3, stride=2, expand_ratio=6),
#                 MBConv2d(80, 80, kernel_size=3, stride=1, expand_ratio=6),
#                 MBConv2d(80, 80, kernel_size=3, stride=1, expand_ratio=6),
#                 # 5
#                 MBConv2d(80, 112, kernel_size=5, stride=2, expand_ratio=6),
#                 MBConv2d(112, 112, kernel_size=5, stride=1, expand_ratio=6),
#                 MBConv2d(112, 112, kernel_size=5, stride=1, expand_ratio=6),
#                 # 6
#                 MBConv2d(112, 192, kernel_size=5, stride=1, expand_ratio=6),
#                 MBConv2d(192, 192, kernel_size=5, stride=1, expand_ratio=6),
#                 MBConv2d(192, 192, kernel_size=5, stride=1, expand_ratio=6),
#                 MBConv2d(192, 192, kernel_size=5, stride=1, expand_ratio=6),
#                 # 7
#                 MBConv2d(192, 320, kernel_size=3, stride=2, expand_ratio=6),
#             ]
#         )

#         super(EfficientNetV1B0, self).__init__(config, layers, 320)
