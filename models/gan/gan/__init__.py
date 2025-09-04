from typing import final, override
import torch
import torch.nn as nn

from core.config import GANConfig

from ..base_model import GANBaseModel


@final
class Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super(Discriminator, self).__init__()

        self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.linear1 = nn.Linear(in_channels, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)

        _ = nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity="relu")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.activation_fn(self.bn1(self.linear1(x)))
        o = self.dropout(o)
        o = self.activation_fn(self.bn2(self.linear2(o)))
        o = self.dropout(o)
        o = self.linear3(o)
        o = o.squeeze(dim=1)

        return o


@final
class Generator(nn.Module):
    def __init__(self, output_size: int = 784):
        super(Generator, self).__init__()

        self.activation_fn = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(100, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.linear4 = nn.Linear(1024, output_size)

        _ = nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity="relu")
        _ = nn.init.kaiming_uniform_(self.linear4.weight, nonlinearity="relu")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.activation_fn(self.bn1(self.linear1(x)))
        o = self.activation_fn(self.bn2(self.linear2(o)))
        o = self.activation_fn(self.bn3(self.linear3(o)))
        o = self.linear4(o)

        return o


@final
class GAN(GANBaseModel):
    def __init__(self, config: GANConfig):
        super(GAN, self).__init__(config)

        g_output_size = (
            784
            if config.dataset == "mnist"
            else 1024
            if config.dataset == "cifar10" or config.dataset == "cifar100"
            else 50176
        )

        self.generator = Generator(output_size=g_output_size)
        self.discriminator = Discriminator(g_output_size)

    # @override
    # def summary(self, input_size: tuple[int, int, int, int]):
    #     g_output_size = (
    #         784
    #         if self.config.dataset == "mnist"
    #         else 1024
    #         if self.config.dataset == "cifar10" or self.config.dataset == "cifar100"
    #         else 50176
    #     )

    #     _ = summary(self.generator, input_size=(1, 100))
    #     _ = summary(
    #         self.discriminator,
    #         input_size=(1, g_output_size),
    #     )
