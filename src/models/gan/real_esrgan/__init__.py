import math
import random
from typing import Literal, final, override

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import GANConfig
from ..base_model import BaseGANModel


@final
class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_norm: bool = True,
    ):
        super(DiscriminatorBlock, self).__init__()

        if use_norm:
            conv = spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=False,
            )

        self.sequence = nn.Sequential(
            conv,
            nn.LeakyReLU(0.2, inplace=True),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.sequence(x)

        return o


@final
class Discriminator(nn.Module):
    def __init__(
        self, in_channels: int = 3, feat: int = 64, skip_connection: bool = True
    ):
        super(Discriminator, self).__init__()

        self.skip_connection = skip_connection

        self.block1 = DiscriminatorBlock(in_channels, feat, 3, 1, False)

        self.block2 = DiscriminatorBlock(feat, feat * 2, 4, 2)
        self.block3 = DiscriminatorBlock(feat * 2, feat * 4, 4, 2)
        self.block4 = DiscriminatorBlock(feat * 4, feat * 8, 3, 2)

        self.block5 = DiscriminatorBlock(feat * 8, feat * 4, 3, 1)
        self.block6 = DiscriminatorBlock(feat * 4, feat * 2, 3, 1)
        self.block7 = DiscriminatorBlock(feat * 2, feat, 3, 1)

        self.block8 = DiscriminatorBlock(feat, feat, 3, 1)
        self.block9 = DiscriminatorBlock(feat, feat, 3, 1)
        self.block10 = DiscriminatorBlock(feat, 1, 3, 1, False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.block1(x)

        o2 = self.block2(o1)
        o3 = self.block3(o2)
        o = self.block4(o3)

        o = F.interpolate(o, scale_factor=2, mode="bilinear", align_corners=False)
        o = self.block5(o)

        if self.skip_connection:
            o = o + o3

        o = F.interpolate(o, scale_factor=2, mode="bilinear", align_corners=False)
        o = self.block6(o)

        if self.skip_connection:
            o = o + o2

        o = F.interpolate(o, scale_factor=2, mode="bilinear", align_corners=False)
        o = self.block7(o)

        if self.skip_connection:
            o = o + o1

        o = self.block8(o)
        o = self.block9(o)
        o = self.block10(o)

        return o


@final
class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, is_last: bool):
        super(DenseLayer, self).__init__()
        self.is_last = is_last

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.lrelu = nn.LeakyReLU(0.2)

    @override
    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = [x]

        o = torch.cat(x, dim=1)

        o = self.conv(o)

        if not self.is_last:
            o = self.lrelu(o)

        return o


@final
class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        for i in range(4):
            self.add_module(
                f"dense_layer_{i + 1}",
                DenseLayer(64 + 32 * i, False),
            )

        self.add_module("dense_layer_5", DenseLayer(192, True))

    @override
    def forward(self, x: torch.Tensor):
        features = [x]

        for layer in self.children():
            o = layer(features)
            features.append(o)

        o = torch.cat(features, dim=1)
        return o


@final
class RRDB(nn.Module):
    def __init__(self, beta: float) -> None:
        super(RRDB, self).__init__()

        self.dense_block1 = DenseBlock()
        self.transition_conv1 = nn.Conv2d(
            in_channels=224, out_channels=64, kernel_size=1
        )
        self.dense_block2 = DenseBlock()
        self.transition_conv2 = nn.Conv2d(
            in_channels=224, out_channels=64, kernel_size=1
        )
        self.dense_block3 = DenseBlock()
        self.transition_conv3 = nn.Conv2d(
            in_channels=224, out_channels=64, kernel_size=1
        )

        self.beta = beta

    @override
    def forward(self, x: torch.Tensor):
        o0 = x

        o = self.dense_block1(x)
        o = self.transition_conv1(o)
        o = o * self.beta
        o = o + o0

        o1 = o

        o = self.dense_block2(o)
        o = self.transition_conv2(o)
        o = o * self.beta
        o = o + o1

        o2 = o

        o = self.dense_block3(o)
        o = self.transition_conv3(o)
        o = o * self.beta
        o = o + o2

        o = o * self.beta

        o = o + o0

        return o


@final
class UpsampleBlock(nn.Module):
    def __init__(self):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.lrelu = nn.LeakyReLU(0.2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv(x)
        o = self.pixel_shuffle(o)
        o = self.lrelu(o)

        return o


@final
class Generator(nn.Module):
    def __init__(self, rrdb_layers: int, beta: float):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False
            ),
            nn.LeakyReLU(0.2),
        )

        self.rrdb = nn.ModuleList([RRDB(beta) for _ in range(rrdb_layers)])

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.upsample_block1 = UpsampleBlock()
        self.upsample_block2 = UpsampleBlock()

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, padding=4, bias=False
        )

    @override
    def forward(self, x: torch.Tensor):
        o = self.layer1(x)
        for layer in self.rrdb:
            o = layer(o)
        o = self.layer2(o)
        o = self.upsample_block1(o)
        o = self.upsample_block2(o)

        o = self.conv3(o)

        return o


def sigma_matrix2(sigma_x: torch.Tensor, sigma_y: torch.Tensor, theta: torch.Tensor):
    d_matrix = torch.tensor([[sigma_x**2, 0], [0, sigma_y**2]], dtype=torch.float32)
    u_matrix = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ],
        dtype=torch.float32,
    )

    return u_matrix @ d_matrix @ u_matrix.T


def mesh_grid(kernel_size: int):
    ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)
    return xy.reshape(-1, 2)


def bivariate_gaussian(
    kernel_size: int,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    theta: torch.Tensor,
    grid: torch.Tensor | None = None,
    isotropic: bool = True,
):
    if grid is None:
        grid = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = torch.tensor(
            [[sigma_x**2, 0], [0, sigma_x**2]], dtype=torch.float32
        )
    else:
        sigma_matrix = sigma_matrix2(sigma_x, sigma_y, theta)

    inverse_sigma = torch.linalg.inv(sigma_matrix)
    kernel = torch.exp(-0.5 * torch.sum(grid @ inverse_sigma * grid, dim=-1))
    kernel /= torch.sum(kernel)

    return kernel.reshape(kernel_size, kernel_size)


def bivariate_generalized_gaussian(
    kernel_size: int,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    theta: torch.Tensor,
    beta: torch.Tensor,
    grid: torch.Tensor | None = None,
    isotropic: bool = True,
):
    """https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/degradations.py"""
    if grid is None:
        grid = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = torch.tensor(
            [[sigma_x**2, 0], [0, sigma_x**2]], dtype=torch.float32
        )
    else:
        sigma_matrix = sigma_matrix2(sigma_x, sigma_y, theta)

    inverse_sigma = torch.linalg.inv(sigma_matrix)
    kernel = torch.exp(
        -0.5 * (torch.sum((grid @ inverse_sigma) * grid, dim=-1)).pow(beta)
    )
    kernel /= torch.sum(kernel)

    return kernel.reshape(kernel_size, kernel_size)


def bivariate_plateau(
    kernel_size: int,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    theta: torch.Tensor,
    beta: torch.Tensor,
    grid: torch.Tensor | None = None,
    isotropic: bool = True,
):
    if grid is None:
        grid = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = torch.tensor(
            [[sigma_x**2, 0], [0, sigma_x**2]], dtype=torch.float32
        )
    else:
        sigma_matrix = sigma_matrix2(sigma_x, sigma_y, theta)

    inverse_sigma = torch.linalg.inv(sigma_matrix)
    kernel = torch.reciprocal(
        torch.sum((grid @ inverse_sigma) * grid, dim=-1).pow(beta) + 1
    )
    kernel /= torch.sum(kernel)

    return kernel.reshape(kernel_size, kernel_size)


def random_bivariate_gaussian(
    kernel_size: int,
    sigma_x_range: tuple[float, float],
    sigma_y_range: tuple[float, float],
    rotation_range: tuple[float, float],
    noise_range: tuple[float, float] | None = None,
    isotropic: bool = True,
):
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    assert sigma_x_range[0] < sigma_x_range[1], "sigma_x must be in ascending order."

    sigma_x = torch.empty(1).uniform_(*sigma_x_range)

    if not isotropic:
        assert sigma_y_range[0] < sigma_y_range[1], (
            "sigma_y must be in ascending order."
        )
        assert rotation_range[0] < rotation_range[1], (
            "rotation must be in ascending order."
        )
        sigma_y = torch.empty(1).uniform_(*sigma_y_range)
        rotation = torch.empty(1).uniform_(*rotation_range)
    else:
        sigma_y = sigma_x
        rotation = torch.Tensor([0.0])

    kernel = bivariate_gaussian(
        kernel_size=kernel_size,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta=rotation,
        isotropic=isotropic,
    )

    if noise_range is not None:
        assert noise_range[0] < noise_range[1], (
            "Noise range must be in ascending order."
        )
        noise = torch.empty(kernel.shape).uniform_(*noise_range).item()
        kernel += noise * torch.randn_like(kernel)

    kernel = kernel / torch.sum(kernel)

    return kernel


def random_bivariate_generalized_gaussian(
    kernel_size: int,
    sigma_x_range: tuple[float, float],
    sigma_y_range: tuple[float, float],
    rotation_range: tuple[float, float],
    beta_range: tuple[float, float],
    noise_range: tuple[float, float] | None = None,
    isotropic: bool = True,
):
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    assert sigma_x_range[0] < sigma_x_range[1], "sigma_x must be in ascending order."
    assert beta_range[0] < beta_range[1], "beta must be in ascending order."

    sigma_x = torch.empty(1).uniform_(*sigma_x_range)

    if not isotropic:
        assert sigma_y_range[0] < sigma_y_range[1], (
            "sigma_y must be in ascending order."
        )
        assert rotation_range[0] < rotation_range[1], (
            "rotation must be in ascending order."
        )
        sigma_y = torch.empty(1).uniform_(*sigma_y_range)
        rotation = torch.empty(1).uniform_(*rotation_range)
    else:
        sigma_y = sigma_x
        rotation = torch.Tensor([0.0])

    if torch.rand(1).item() < 0.5:
        beta = torch.empty(1).uniform_(beta_range[0], 1)
    else:
        beta = torch.empty(1).uniform_(1, beta_range[1])

    kernel = bivariate_generalized_gaussian(
        kernel_size=kernel_size,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta=rotation,
        beta=beta,
        isotropic=isotropic,
    )

    if noise_range is not None:
        assert noise_range[0] < noise_range[1], (
            "Noise range must be in ascending order."
        )
        noise = torch.empty(kernel.shape).uniform_(*noise_range).item()
        kernel += noise * torch.randn_like(kernel)

    kernel = kernel / torch.sum(kernel)

    return kernel


def random_bivariate_plateau(
    kernel_size: int,
    sigma_x_range: tuple[float, float],
    sigma_y_range: tuple[float, float],
    rotation_range: tuple[float, float],
    beta_range: tuple[float, float],
    noise_range: tuple[float, float] | None = None,
    isotropic: bool = True,
):
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    assert sigma_x_range[0] < sigma_x_range[1], "sigma_x must be in ascending order."
    assert beta_range[0] < beta_range[1], "beta must be in ascending order."

    sigma_x = torch.empty(1).uniform_(*sigma_x_range)

    if not isotropic:
        assert sigma_y_range[0] < sigma_y_range[1], (
            "sigma_y must be in ascending order."
        )
        assert rotation_range[0] < rotation_range[1], (
            "rotation must be in ascending order."
        )
        sigma_y = torch.empty(1).uniform_(*sigma_y_range)
        rotation = torch.empty(1).uniform_(*rotation_range)
    else:
        sigma_y = sigma_x
        rotation = torch.Tensor([0.0])

    if torch.rand(1).item() < 0.5:
        beta = torch.empty(1).uniform_(beta_range[0], 1)
    else:
        beta = torch.empty(1).uniform_(1, beta_range[1])

    kernel = bivariate_plateau(
        kernel_size=kernel_size,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta=rotation,
        beta=beta,
        isotropic=isotropic,
    )

    if noise_range is not None:
        assert noise_range[0] < noise_range[1], (
            "Noise range must be in ascending order."
        )
        noise = torch.empty(kernel.shape).uniform_(*noise_range).item()
        kernel += noise * torch.randn_like(kernel)

    kernel = kernel / torch.sum(kernel)

    return kernel


kernel_list_type = Literal[
    "iso",
    "aniso",
    "generalized_iso",
    "generalized_aniso",
    "plateau_iso",
    "plateau_aniso",
]


def random_mixed_kernels(
    image: torch.Tensor,
    kernel_list: list[kernel_list_type],
    kernel_prob: list[float],
    kernel_size: int,
    sigma_x_range: tuple[float, float],
    sigma_y_range: tuple[float, float],
    rotation_range: tuple[float, float],
    betag_range: tuple[float, float],
    betap_range: tuple[float, float],
    noise_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    assert len(kernel_list) == len(kernel_prob), (
        "Kernel list and probability list must have the same length."
    )
    assert math.isclose(sum(kernel_prob), 1.0), "Kernel probabilities must sum to 1."

    kernel_type = random.choices(kernel_list, weights=kernel_prob, k=1)[0]

    if kernel_type == "iso":
        kernel = random_bivariate_gaussian(
            kernel_size=kernel_size,
            sigma_x_range=sigma_x_range,
            sigma_y_range=sigma_x_range,
            rotation_range=rotation_range,
            noise_range=noise_range,
            isotropic=True,
        )
    elif kernel_type == "aniso":
        kernel = random_bivariate_gaussian(
            kernel_size=kernel_size,
            sigma_x_range=sigma_x_range,
            sigma_y_range=sigma_y_range,
            rotation_range=rotation_range,
            noise_range=noise_range,
            isotropic=False,
        )
    elif kernel_type == "generalized_iso":
        kernel = random_bivariate_generalized_gaussian(
            kernel_size=kernel_size,
            sigma_x_range=sigma_x_range,
            sigma_y_range=sigma_y_range,
            rotation_range=rotation_range,
            beta_range=betag_range,
            noise_range=noise_range,
            isotropic=True,
        )
    elif kernel_type == "generalized_aniso":
        kernel = random_bivariate_generalized_gaussian(
            kernel_size=kernel_size,
            sigma_x_range=sigma_x_range,
            sigma_y_range=sigma_y_range,
            rotation_range=rotation_range,
            beta_range=betag_range,
            noise_range=noise_range,
            isotropic=False,
        )
    elif kernel_type == "plateau_iso":
        kernel = random_bivariate_plateau(
            kernel_size=kernel_size,
            sigma_x_range=sigma_x_range,
            sigma_y_range=sigma_x_range,
            rotation_range=rotation_range,
            beta_range=betap_range,
            noise_range=noise_range,
            isotropic=True,
        )
    else:  # kernel_type == "plateau_aniso"
        kernel = random_bivariate_plateau(
            kernel_size=kernel_size,
            sigma_x_range=sigma_x_range,
            sigma_y_range=sigma_y_range,
            rotation_range=rotation_range,
            beta_range=betap_range,
            noise_range=noise_range,
            isotropic=False,
        )

    padding = (21 - kernel_size) // 2
    kernel = F.pad(
        kernel, (padding, padding, padding, padding), mode="constant", value=0
    )
    kernel = (
        kernel.unsqueeze(0).repeat(image.shape[0], 1, 1).to(image.device, image.dtype)
    )

    k = kernel.size(-1)
    b, c, h, w = image.shape

    o = F.pad(image, (k // 2, k // 2, k // 2, k // 2), mode="reflect")

    ph, pw = o.size()[-2:]

    if kernel.size(0) == 1:
        o = o.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(o, kernel, padding=0).view(b, c, h, w)
    else:
        o = o.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(o, kernel, groups=b * c).view(b, c, h, w)


resize_type = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
]


def random_resize(
    image: torch.Tensor,
    resize_types: list[resize_type],
    resize_prob: list[float],
) -> torch.Tensor:
    assert len(resize_types) == len(resize_prob), (
        "Resize types and probability list must have the same length."
    )
    assert math.isclose(sum(resize_prob), 1.0), "Resize probabilities must sum to 1."

    resize_type = random.choices(resize_types, weights=resize_prob, k=1)[0]
    # linear', 'bilinear', 'bicubic' or 'trilinear'.
    match resize_type:
        case "nearest":
            o = F.interpolate(image, scale_factor=0.5, mode="nearest")
        case "linear":
            o = F.interpolate(
                image, scale_factor=0.5, mode="linear", align_corners=False
            )
        case "bilinear":
            o = F.interpolate(
                image, scale_factor=0.5, mode="bilinear", align_corners=False
            )
        case "bicubic":
            o = F.interpolate(
                image, scale_factor=0.5, mode="bicubic", align_corners=False
            )
        case "trilinear":
            o = F.interpolate(
                image, scale_factor=0.5, mode="trilinear", align_corners=False
            )
        case "area":
            o = F.interpolate(image, scale_factor=0.5, mode="area")
        case "nearest-exact":
            o = F.interpolate(image, scale_factor=0.5, mode="nearest-exact")

    return o.clamp(0, 1)


def gray_noise(
    image: torch.Tensor, mean: float = 0.0, sigma: float = 1.0
) -> torch.Tensor:
    noise = torch.randn_like(image) * sigma + mean
    noise = noise.to(image.device)
    noise = noise.clamp(0, 1)
    return image + noise


noise_type = Literal["gaussian", "poisson"]


def random_noise(
    image: torch.Tensor,
    noise_types: list[noise_type],
    noise_prob: list[float],
    mean: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.01,
) -> torch.Tensor:
    assert len(noise_types) == len(noise_prob), (
        "Noise types and probability list must have the same length."
    )
    assert math.isclose(sum(noise_prob), 1.0), "Noise probabilities must sum to 1."

    noise_type = random.choices(noise_types, weights=noise_prob, k=1)[0]

    match noise_type:
        case "gaussian":
            o = transforms.GaussianNoise(mean=mean, sigma=sigma)(image)
        case "poisson":
            noise = alpha * (torch.poisson(image / alpha) - (image / alpha))
            o = image + noise
            o = o.clamp(0, 1)

    if np.random.rand() < 0.4:
        o = gray_noise(o, mean=mean, sigma=sigma)

    return o


@final
class Degrader1(nn.Module):
    def __init__(self, image_size: int, poisson_alpha: float):
        super(Degrader1, self).__init__()

        self.image_size = image_size
        self.poisson_alpha = poisson_alpha

        self.jpeg = transforms.JPEG([30, 95])

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = random_mixed_kernels(
            x,
            kernel_list=[
                "iso",
                "aniso",
                "generalized_iso",
                "generalized_aniso",
                "plateau_iso",
                "plateau_aniso",
            ],
            kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            kernel_size=21,
            sigma_x_range=(0.2, 3),
            sigma_y_range=(0.2, 3),
            rotation_range=(-math.pi, math.pi),
            betag_range=(0.5, 4.0),
            betap_range=(1.0, 2.0),
            noise_range=None,
        )

        if np.random.rand() < 0.1:
            o = torch.sinc(o)

        o = random_resize(
            o,
            resize_types=["nearest", "area", "bilinear", "bicubic"],
            resize_prob=[0.25, 0.25, 0.25, 0.25],
        )

        o = random_noise(
            o,
            noise_types=["gaussian", "poisson"],
            noise_prob=[0.5, 0.5],
            mean=0.0,
            sigma=0.05,
            alpha=self.poisson_alpha,
        )

        o = o * 255
        o = o.to(torch.uint8)
        o = self.jpeg(o) / 255.0

        return o


@final
class Degrader2(nn.Module):
    def __init__(self, image_size: int, poisson_alpha: float):
        super(Degrader2, self).__init__()

        self.image_size = image_size
        self.poisson_alpha = poisson_alpha

        self.jpeg = transforms.JPEG([30, 95])

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = random_mixed_kernels(
            x,
            kernel_list=[
                "iso",
                "aniso",
                "generalized_iso",
                "generalized_aniso",
                "plateau_iso",
                "plateau_aniso",
            ],
            kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
            kernel_size=21,
            sigma_x_range=(0.2, 1.5),
            sigma_y_range=(0.2, 1.5),
            rotation_range=(-math.pi, math.pi),
            betag_range=(0.5, 4.0),
            betap_range=(1.0, 2.0),
            noise_range=None,
        )

        o = random_resize(
            o,
            resize_types=["nearest", "area", "bilinear", "bicubic"],
            resize_prob=[0.25, 0.25, 0.25, 0.25],
        )

        o = random_noise(
            o,
            noise_types=["gaussian", "poisson"],
            noise_prob=[0.5, 0.5],
            mean=0.0,
            sigma=0.05,
            alpha=self.poisson_alpha,
        )

        o = o * 255.0
        o = o.to(torch.uint8)
        o = self.jpeg(o) / 255.0

        if np.random.rand() < 0.8:
            o = torch.sinc(o)

        return o


@final
class RealESRGAN(BaseGANModel):
    def __init__(self, config: GANConfig):
        super(RealESRGAN, self).__init__(config)

        poisson_alpha = config.model_params.get("poisson_alpha", 0.01)

        beta: float = config.model_params["beta"] or 1.0
        assert 0 < beta <= 1

        rrdb_layers: int = config.model_params["rrdb_layers"] or 16
        assert 0 < rrdb_layers

        image_size = config.model_params.get("image_size", None)

        if config.dataset == "df2k_ost":
            in_channels = 3
            if image_size is None:
                image_size = 256
        elif config.dataset == "df2k_ost_small":
            in_channels = 3
            if image_size is None:
                image_size = 96
        else:
            in_channels = 3
            if image_size is None:
                image_size = 64

        self.degrader1 = Degrader1(image_size, poisson_alpha=poisson_alpha)
        self.degrader2 = Degrader2(image_size // 2, poisson_alpha=poisson_alpha)

        self.discriminator = Discriminator(in_channels)
        self.generator = Generator(rrdb_layers, beta)

    @override
    def pretrain_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        _ = self.generator.train()
        _ = self.generator.to(self.device)

        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc="Pretraining"):
            z, targets = batch
            z = z.to(self.device)
            targets = targets.to(self.device)

            z = self.degrader1(z)
            z = self.degrader2(z)

            optimizer.zero_grad()

            g_z = self.generator(z)

            g_loss = loss_function(g_z, targets)

            g_loss.backward()
            optimizer.step()

            epoch_loss += g_loss.item()

        epoch_loss /= len(train_loader)

        _ = self.generator.train(False)

        return epoch_loss

    @override
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        g_loss_function: nn.Module,
        d_loss_function: nn.Module,
    ) -> tuple[float, float]:
        """Train the model."""
        _ = self.generator.train()
        _ = self.discriminator.train()
        _ = self.degrader1.to(self.device)
        _ = self.degrader2.to(self.device)
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            z, targets = batch
            z = z.to(self.device)
            targets = targets.to(self.device)

            #######################
            #  Preprocess inputs  #
            #######################

            z = self.degrader1(z)
            z = self.degrader2(z)

            #######################
            # Train Discriminator #
            #######################

            d_optimizer.zero_grad()

            d_x = self.discriminator(targets)
            g_z = self.generator(z)
            d_g_z = self.discriminator(g_z.detach())

            d_loss = d_loss_function(d_x, d_g_z)

            d_loss.backward()
            d_optimizer.step()

            #######################
            #   Train Generator   #
            #######################

            g_optimizer.zero_grad()

            d_g_z = self.discriminator(g_z)

            g_loss = g_loss_function(targets, g_z, d_g_z)

            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        _ = self.generator.train(False)
        _ = self.discriminator.train(False)

        return epoch_g_loss / len(train_loader), epoch_d_loss / len(train_loader)

    @override
    def predict(self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]):
        _ = self.generator.eval()

        with torch.no_grad():
            batch = next(iter(data_loader))
            z, targets = batch

            g_z = self.generator(z.to(self.device)).view(-1, 3, 96, 96)
            g_z = g_z.clamp(0, 1)  # Ensure pixel values are in [0, 1]
            g_z = g_z * 255.0  # Scale to [0, 255]
            g_z = g_z.byte()  # Convert to byte format

            generated_images = g_z.cpu()

        # show all generated_images and targets
        _, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(transforms.ToPILImage()(targets[i].cpu()), aspect="auto")
            axes[0, i].axis("off")
            axes[1, i].imshow(
                transforms.ToPILImage()(generated_images[i]), aspect="auto"
            )
            axes[1, i].axis("off")
        axes[0, 0].set_title("Original")
        axes[1, 0].set_title("Generated")
        plt.tight_layout()
        plt.show()
