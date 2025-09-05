import itertools
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
from core.registry import ModelRegistry
from ..base_model import GANBaseModel


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


# JPEG Noise
# https://madbrain.ai/implementing-jpeg-compression-in-pytorch-b0a830889f59


@final
class ColorConversion(nn.Module):
    """
    Module to convert RGB images to YCbCr color space.
    """

    def __init__(self):
        """
        Initializes the ColorConversion module with the RGB to YCbCr conversion matrix and shift.
        """
        super().__init__()

        # Define the RGB to YCbCr conversion matrix
        self.rgb_to_ycbcr_matrix = torch.tensor(
            [
                [0.2990, 0.5870, 0.1140],  # Y channel coefficients
                [-0.168736, -0.331264, 0.5],  # Cb channel coefficients
                [0.5, -0.418688, -0.081312],  # Cr channel coefficients
            ],
            dtype=torch.float32,
        )

        # Define the shift to be added after the matrix multiplication
        self.shift = torch.tensor([0, 128, 128], dtype=torch.float32).view(1, 3, 1, 1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts an RGB image tensor to YCbCr color space.

        Args:
                x (torch.Tensor): Input image tensor of shape (B, C, H, W) with RGB channels.

        Returns:
                torch.Tensor: YCbCr image tensor of shape (B, C, H, W).
        """

        # Perform matrix multiplication between the RGB channels and the conversion matrix
        # 'ij,njhw->nihw' specifies the Einstein summation for batch processing
        print(self.rgb_to_ycbcr_matrix.shape)
        print(x.shape)
        o = torch.einsum(
            "ij,njhw->nihw", self.rgb_to_ycbcr_matrix.to(x.device), x
        ) + self.shift.to(x.device)

        return o


@final
class InvertColorConversion(nn.Module):
    """
    Module to convert YCbCr images back to RGB color space.
    """

    def __init__(self):
        """
        Initializes the InvertColorConversion module with the YCbCr to RGB conversion matrix and shift.
        """
        super().__init__()

        # Define the YCbCr to RGB conversion matrix
        self.ycbcr_to_rgb_matrix = torch.tensor(
            [
                [1.0, 0.0, 1.4020],  # R channel coefficients
                [1.0, -0.344136, -0.714136],  # G channel coefficients
                [1.0, 1.7720, 0.0],  # B channel coefficients
            ],
            dtype=torch.float32,
        )

        # Define the shift to be subtracted before the matrix multiplication
        self.shift = torch.tensor([0, 128, 128], dtype=torch.float32).view(1, 3, 1, 1)

    @override
    def forward(self, ycbcr: torch.Tensor):
        """
        Converts a YCbCr image tensor back to RGB color space.

        Args:
                ycbcr (torch.Tensor): Input YCbCr image tensor of shape (B, C, H, W).

        Returns:
                torch.Tensor: RGB image tensor of shape (B, C, H, W) with values clamped between 0 and 255.
        """

        # Subtract the shift to center the Cb and Cr channels
        o = ycbcr - self.shift.to(ycbcr.device)

        # Perform matrix multiplication between the YCbCr channels and the inverse conversion matrix
        o = torch.einsum("ij,njhw->nihw", self.inv_matrix.to(ycbcr.device), o)

        # Clamp the RGB values to ensure they are within the valid range [0, 255]
        o = torch.clamp(o, 0, 255)

        return o


@final
class ChromaDownsample(nn.Module):
    """
    Module to perform chroma subsampling on YCbCr images.

    Chroma subsampling reduces the resolution of the Cb and Cr channels
    to decrease the amount of color information, which is a common
    technique in JPEG compression to exploit the human eye's lower
    sensitivity to color details.
    """

    def __init__(self, factor: Literal[1, 2, 4] = 2):
        """
        Initializes the ChromaDownsample module.

        Args:
                factor (int, optional): Subsampling factor. Must be 1, 2, or 4. Defaults to 2.
        """
        super().__init__()

        self.factor = factor  # Subsampling factor

    @override
    def forward(
        self, ycbcr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs chroma subsampling on the input YCbCr image.

        Args:
                ycbcr (torch.Tensor): Input image tensor in YCbCr color space
                                                          with shape (B, 3, H, W).

        Returns:
                tuple:
                        - Y (torch.Tensor): Luminance channel with shape (B, 1, H, W).
                        - Cb_subsampled (torch.Tensor): Subsampled Cb channel.
                        - Cr_subsampled (torch.Tensor): Subsampled Cr channel.
        """

        # Extract the Y, Cb, and Cr channels from the input tensor
        Y = ycbcr[:, 0:1, :, :]  # Luminance channel
        Cb = ycbcr[:, 1:2, :, :]  # Blue-difference chroma channel
        Cr = ycbcr[:, 2:3, :, :]  # Red-difference chroma channel

        if self.factor > 1:
            # Apply average pooling to subsample Cb and Cr channels
            # Reduces the spatial resolution by the specified factor
            Cb_subsampled = F.avg_pool2d(
                Cb, kernel_size=self.factor, stride=self.factor
            )
            Cr_subsampled = F.avg_pool2d(
                Cr, kernel_size=self.factor, stride=self.factor
            )
        else:
            # If factor is 1, no subsampling is performed
            Cb_subsampled = Cb
            Cr_subsampled = Cr

        return Y, Cb_subsampled, Cr_subsampled


@final
class ChromaUpsample(nn.Module):
    """
    Module to perform chroma upsampling on YCbCr images.

    Chroma upsampling restores the resolution of the Cb and Cr channels
    to match the luminance channel, reversing the chroma subsampling step.
    """

    def __init__(self, factor: Literal[1, 2, 4] = 2):
        """
        Initializes the ChromaUpsample module.

        Args:
                factor (int, optional): Upsampling factor. Must be 1, 2, or 4. Default is 2.
        """
        super().__init__()

        self.factor = factor  # Upsampling factor

    @override
    def forward(self, Cb: torch.Tensor, Cr: torch.Tensor):
        """
        Performs chroma upsampling on the Cb and Cr channels.

        Args:
                Cb (torch.Tensor): Subsampled Cb channel with shape (B, 1, H', W').
                Cr (torch.Tensor): Subsampled Cr channel with shape (B, 1, H', W').

        Returns:
                tuple:
                        - Cb_upsampled (torch.Tensor): Upsampled Cb channel with shape (B, 1, H, W).
                        - Cr_upsampled (torch.Tensor): Upsampled Cr channel with shape (B, 1, H, W).
        """

        if self.factor > 1:
            # Use bilinear interpolation to upsample Cb and Cr channels
            # align_corners=False ensures smoother upsampling
            Cb_upsampled = F.interpolate(
                Cb, scale_factor=self.factor, mode="bilinear", align_corners=False
            )
            Cr_upsampled = F.interpolate(
                Cr, scale_factor=self.factor, mode="bilinear", align_corners=False
            )
        else:
            # If factor is 1, no upsampling is performed
            Cb_upsampled = Cb
            Cr_upsampled = Cr

        return Cb_upsampled, Cr_upsampled


@final
class BlockSplitting(nn.Module):
    """
    Module to split an image into non-overlapping blocks.

    This is useful in JPEG compression where the image is divided into
    blocks (commonly 8x8) for processing such as DCT transformation.
    """

    def __init__(self, block_size: int = 8):
        """
        Initializes the BlockSplitting module.

        Args:
                block_size (int, optional): Size of each block. Default is 8.
        """
        super().__init__()

        self.block_size = block_size

    @override
    def forward(self, x: torch.Tensor):
        """
        Splits the input image into non-overlapping blocks.

        Args:
                x (torch.Tensor): Input image tensor of shape (B, C, H, W),

        Returns:
                torch.Tensor: Tensor containing image blocks with shape (B, num_blocks * C, block_size, block_size),
                                          where num_blocks = (H // block_size) * (W // block_size).
        """
        B, _, H, W = x.shape  # Batch size, Channels, Height, Width

        # Ensure that the image dimensions are divisible by the block size
        if H % self.block_size != 0 or W % self.block_size != 0:
            raise ValueError(
                f"Image dimensions ({H}x{W}) are not divisible by block size {self.block_size}."
            )

        # Reshape the image to separate blocks
        o = x.view(B, H // self.block_size, self.block_size, -1, self.block_size)

        # Permute dimensions to bring block dimensions together
        o = o.permute(0, 1, 3, 2, 4)

        # Combine the batch and block grid dimensions, and merge channels with blocks
        o = o.contiguous().view(B, -1, self.block_size, self.block_size)

        return o


@final
class BlockMerging(nn.Module):
    """
    Module to merge non-overlapping blocks back into the original image.

    This reverses the BlockSplitting process, reconstructing the full image from its blocks.
    """

    def __init__(self, block_size: int = 8):
        """
        Initializes the BlockMerging module.

        Args:
                block_size (int, optional): Size of each block. Default is 8.
        """
        super().__init__()

        self.block_size = block_size

    @override
    def forward(self, x: torch.Tensor, original_size: tuple[int, int]):
        """
        Merges blocks to reconstruct the original image.

        Args:
                x (torch.Tensor): Tensor containing image blocks with shape (B, num_blocks * C, block_size, block_size),
                                                           where num_blocks = (H // block_size) * (W // block_size).
                original_size (tuple): Tuple containing the original image dimensions as (H, W).

        Returns:
                torch.Tensor: Reconstructed image tensor of shape (B, C, H, W).
        """
        H, W = original_size  # Original Height and Width
        B = x.shape[0]  # Batch size

        # Reshape blocks to separate channels and block grid
        o = x.view(
            B,
            H // self.block_size,
            W // self.block_size,
            self.block_size,
            self.block_size,
        )

        # Permute dimensions to arrange blocks correctly
        o = o.permute(0, 1, 3, 2, 4)

        # Merge the block dimensions to reconstruct the full image
        o = o.contiguous().view(B, 1, H, W)

        return o


@final
class DCT2D(nn.Module):
    """
    Module to perform the 2D Discrete Cosine Transform (DCT) on image blocks.

    The DCT is used in JPEG compression to convert spatial domain data into frequency domain,
    enabling efficient compression by focusing on the most significant frequency components.
    """

    def __init__(self, block_size: int = 8):
        """
        Initializes the DCT2D module by creating the DCT basis tensor and scaling factors.

        Args:
                block_size (int, optional): Size of the blocks to perform DCT on. Default is 8.
        """
        super().__init__()

        # Initialize an empty tensor for the DCT basis functions
        dct_matrix = np.zeros(
            (block_size, block_size, block_size, block_size), dtype=np.float32
        )

        # Compute the DCT basis functions for each combination of spatial and frequency coordinates
        for x, y, u, v in itertools.product(range(block_size), repeat=4):
            dct_matrix[x, y, u, v] = np.cos(
                (2 * x + 1) * u * np.pi / (2 * block_size)
            ) * np.cos((2 * y + 1) * v * np.pi / (2 * block_size))

        self.tensor = torch.from_numpy(dct_matrix)

        # Compute the scaling factors (alpha) for normalization
        alpha = np.ones(block_size, dtype=np.float32)
        alpha[0] = 1 / np.sqrt(2)  # Special scaling for the DC component

        # Compute the outer product of alpha vectors and scale by 0.25 as part of the DCT normalization
        self.scale = torch.from_numpy(np.outer(alpha, alpha) * 0.25)

    @override
    def forward(self, x: torch.Tensor):
        """
        Applies the 2D DCT to the input image blocks.

        Args:
                blocks (torch.Tensor): Input image blocks tensor of shape (B, C, H, W),
                                                           where B is batch size, C is number of channels,
                                                           and H, W are height and width of the blocks (typically 8x8).

        Returns:
                torch.Tensor: DCT-transformed blocks of shape (B, C, H, W).
        """

        # Center the pixel values around zero by subtracting 128 (common in JPEG)
        o = x - 128.0

        # Perform the DCT using tensor dot product
        o = self.scale.to(x.device) * torch.tensordot(
            o, self.tensor.to(x.device), dims=2
        )

        return o


@final
class IDCT2D(nn.Module):
    """
    Module to perform the Inverse 2D Discrete Cosine Transform (IDCT) on DCT coefficients.

    The IDCT is used in JPEG decompression to convert frequency domain data back into spatial domain,
    reconstructing the original image blocks.
    """

    def __init__(self, block_size: int = 8):
        """
        Initializes the IDCT2D module by creating the inverse DCT basis tensor and scaling factors.

        Args:
                block_size (int, optional): Size of the blocks to perform IDCT on. Default is 8.
        """
        super().__init__()

        # Compute the scaling factors (alpha) for normalization
        alpha = np.ones(block_size, dtype=np.float32)
        alpha[0] = 1 / np.sqrt(2)  # Special scaling for the DC component

        # Compute the outer product of alpha vectors for normalization
        self.alpha = torch.from_numpy(np.outer(alpha, alpha))

        # Initialize an empty tensor for the inverse DCT basis functions
        idct_matrix = np.zeros(
            (block_size, block_size, block_size, block_size), dtype=np.float32
        )

        # Compute the inverse DCT basis functions for each combination of frequency and spatial coordinates
        for x, y, u, v in itertools.product(range(block_size), repeat=4):
            idct_matrix[x, y, u, v] = np.cos(
                (2 * u + 1) * x * np.pi / (2 * block_size)
            ) * np.cos((2 * v + 1) * y * np.pi / (2 * block_size))

        self.tensor = torch.from_numpy(idct_matrix)

    @override
    def forward(self, x: torch.Tensor):
        """
        Applies the inverse 2D DCT to the input DCT coefficients.

        Args:
                x (torch.Tensor): DCT coefficients tensor of shape (B, C, H, W),

        Returns:
                torch.Tensor: Reconstructed image blocks of shape (B, C, H, W) with pixel values in [0, 255].
        """
        # Apply the scaling factors to the DCT coefficients for normalization
        o = x * self.alpha.to(x.device)

        # Perform the inverse DCT using tensor dot product
        o = 0.25 * torch.tensordot(o, self.tensor, dims=2)

        # Shift the pixel values back by adding 128 to return to the original range
        o = o + 128.0

        return o


def differentiable_round(x: torch.Tensor):
    """
    Performs a soft (differentiable) rounding of the input tensor.

    This function approximates the standard rounding operation with a differentiable
    alternative, which is useful in scenarios where gradients need to flow through
    the rounding step (e.g., during training neural networks for tasks like compression).

    Instead of using a hard rounding function like `torch.round(x)`, which has zero gradients
    almost everywhere and is thus non-differentiable, this function adds a smooth cubic
    term to enable gradient computation.

    Args:
            x (torch.Tensor): Input tensor to be softly rounded.

    Returns:
            torch.Tensor: Softly rounded tensor with gradients enabled.

    Example:
            >>> x = torch.tensor([1.2, 2.5, 3.7], requires_grad=True)
            >>> y = differentiable_round(x)
            >>> y
            tensor([ 1.2000,  2.5000,  3.7000], grad_fn=<AddBackward0>)
    """

    # Compute the nearest integer to each element in the input tensor
    rounded = torch.round(x)

    # Calculate the difference between the input and its rounded value
    difference = x - rounded

    # Apply a cubic adjustment to the difference to create a smooth transition
    # This makes the rounding operation differentiable by ensuring the gradient
    # is non-zero and varies smoothly around each integer boundary
    smooth_adjustment = difference**3

    # Combine the rounded value with the smooth adjustment
    soft_rounded = rounded + smooth_adjustment

    return soft_rounded


# Quantization Tables for JPEG Compression
# These tables are used to quantize the Discrete Cosine Transform (DCT) coefficients
# during the compression process. Quantization reduces the precision of the DCT
# coefficients, which leads to data compression by eliminating less visually
# significant information.

# Luminance Quantization Table (QY)
# This table is applied to the Y (luminance) channel of the YCbCr color space.
# It prioritizes preserving brightness information, which is more noticeable to the human eye.

QY = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)


# Chroma Quantization Table (QC)
# This table is applied to the Cb and Cr (chrominance) channels of the YCbCr color space.
# It allows for more compression in color information, exploiting the human eye's
# lower sensitivity to color variations compared to brightness.

QC = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def quality_to_factor(quality: int):
    """
    Converts a JPEG quality factor (1-100) to a scaling factor used for quantization tables.

    Args:
            quality (float): JPEG quality factor, in the range [1, 100].

    Returns:
            float: Scaled quantization factor used to adjust quantization tables.
    """

    # Validate the input quality factor
    if not (1 <= quality <= 100):
        raise ValueError("Quality factor must be in the range [1, 100].")

    # Calculate the scaling factor based on the quality factor
    if quality < 50:
        # For lower quality factors, use a higher scaling factor to increase compression
        factor = 5000.0 / quality
    else:
        # For higher quality factors, use a lower scaling factor to reduce compression
        factor = 200.0 - quality * 2.0

    # Normalize the scaling factor
    scaling_factor = factor / 100.0

    return scaling_factor


@final
class Quantization(nn.Module):
    """
    Module to perform quantization on DCT-transformed image blocks.

    Quantization reduces the precision of the DCT coefficients by dividing them
    by a quantization table and applying a differentiable rounding function.
    This step is crucial for achieving compression by eliminating less significant
    frequency components.
    """

    def __init__(self, q_table: torch.Tensor, factor: float = 0.5):
        """
        Initializes the Quantization module.

        Args:
                q_table (torch.Tensor): Quantization table used to scale the DCT coefficients.
                factor (float, optional): Scaling factor to adjust the quantization strength.
                                                                  A higher factor results in more aggressive quantization.
                                                                  Default is 0.5.
        """
        super().__init__()

        # Register the quantization table as a buffer to ensure it's moved to the appropriate device
        self.q_table = q_table

        # Add a small epsilon to the factor to prevent division by zero
        self.factor = factor + 1e-4

    @override
    def forward(self, x: torch.Tensor):
        """
        Applies quantization to the input DCT blocks.

        Args:
                blocks (torch.Tensor): DCT-transformed image blocks with shape (B, C, H, W),

        Returns:
                torch.Tensor: Quantized blocks with the same shape as input, using differentiable rounding.
        """

        # Scale the DCT coefficients by dividing by the quantization table and scaling factor
        o = x.float() / (self.q_table * self.factor)

        # Apply a differentiable rounding function to obtain quantized coefficients
        o = differentiable_round(o)

        return o


@final
class Dequantization(nn.Module):
    """
    Module to perform dequantization on quantized DCT blocks.

    Dequantization restores the scale of the DCT coefficients by multiplying them
    with the quantization table and the scaling factor. This step is essential for
    reconstructing the image during decompression.
    """

    def __init__(self, q_table: torch.Tensor, factor: float = 0.5):
        """
        Initializes the Dequantization module.

        Args:
                q_table (torch.Tensor): Quantization table used to scale the DCT coefficients.
                                                                 Should be the same table used during quantization.
                factor (float, optional): Scaling factor used during quantization to adjust the
                                                                  dequantization strength. Must match the factor used
                                                                  in the Quantization module. Default is 0.5.
        """
        super().__init__()

        # Register the quantization table as a buffer to ensure it's moved to the appropriate device
        self.q_table = q_table

        # Add a small epsilon to the factor to prevent potential numerical issues
        self.factor = factor + 1e-4

    @override
    def forward(self, x: torch.Tensor):
        """
        Applies dequantization to the input quantized DCT blocks.

        Args:
                quantized_blocks (torch.Tensor): Quantized DCT coefficients with shape (B, C, H, W).

        Returns:
                torch.Tensor: Dequantized blocks with the same shape as input, restoring the original scale.
        """

        # Multiply the quantized coefficients by the quantization table and scaling factor
        o = x * (self.q_table * self.factor)
        return o


@final
class ZigZagOrder(nn.Module):
    """
    Module to perform ZigZag ordering on image blocks.

    In JPEG compression, ZigZag ordering is used to rearrange the 8x8 block of DCT coefficients
    into a 1D array. This arrangement groups low-frequency coefficients at the beginning
    and high-frequency coefficients at the end, which is beneficial for subsequent
    compression steps like run-length encoding.
    """

    def __init__(self):
        """
        Initializes the ZigZagOrder module by defining the ZigZag index order.
        """
        super().__init__()

        # Define the ZigZag order as per the JPEG standard for an 8x8 block
        self.index_order = torch.tensor(
            [
                [0, 1, 5, 6, 14, 15, 27, 28],
                [2, 4, 7, 13, 16, 26, 29, 42],
                [3, 8, 12, 17, 25, 30, 41, 43],
                [9, 11, 18, 24, 31, 40, 44, 53],
                [10, 19, 23, 32, 39, 45, 52, 54],
                [20, 22, 33, 38, 46, 51, 55, 60],
                [21, 34, 37, 47, 50, 56, 59, 61],
                [35, 36, 48, 49, 57, 58, 62, 63],
            ],
            dtype=torch.int64,
        ).flatten()

    @override
    def forward(self, x: torch.Tensor):
        """
        Applies ZigZag ordering to the input image blocks.

        Args:
                blocks (torch.Tensor): Input image blocks with shape (B, num_blocks, block_size, block_size),

        Returns:
                torch.Tensor: ZigZag ordered coefficients with shape (B, num_blocks, 64),
                                          where 64 corresponds to the flattened 8x8 block.
        """

        # Extract the dimensions of the input blocks
        B, num_blocks, _, _ = x.shape

        # Flatten the last two dimensions (block_size x block_size) into a single dimension
        o = x.view(B, num_blocks, -1)

        # Apply ZigZag ordering using the predefined index_order
        o = o[..., self.index_order]

        return o


@final
class InverseZigZagOrder(nn.Module):
    """
    Module to perform Inverse ZigZag ordering on ZigZag ordered coefficients.

    This module reconstructs the original 8x8 block of DCT coefficients from the
    1D ZigZag ordered array. It is used during the decoding process to revert
    the ZigZag ordering applied during encoding.
    """

    def __init__(self):
        """
        Initializes the InverseZigZagOrder module by defining both the ZigZag
        index order and its inverse for reconstruction.
        """
        super().__init__()

        # Define the ZigZag order as per the JPEG standard for an 8x8 block
        self.index_order = torch.tensor(
            [
                [0, 1, 5, 6, 14, 15, 27, 28],
                [2, 4, 7, 13, 16, 26, 29, 42],
                [3, 8, 12, 17, 25, 30, 41, 43],
                [9, 11, 18, 24, 31, 40, 44, 53],
                [10, 19, 23, 32, 39, 45, 52, 54],
                [20, 22, 33, 38, 46, 51, 55, 60],
                [21, 34, 37, 47, 50, 56, 59, 61],
                [35, 36, 48, 49, 57, 58, 62, 63],
            ],
            dtype=torch.int64,
        ).flatten()

        # Create the inverse ZigZag index order to map back to the original block
        inverse_index = torch.empty_like(self.index_order)
        inverse_index[self.index_order] = torch.arange(
            len(self.index_order), dtype=torch.int64
        )

        self.inverse_index = inverse_index

    @override
    def forward(self, x: torch.Tensor):
        """
        Applies Inverse ZigZag ordering to the input coefficients.

        Args:
                zigzag_coeffs (torch.Tensor): ZigZag ordered coefficients with shape (B, num_blocks, 64),
                                                                         where 64 corresponds to the flattened 8x8 block.

        Returns:
                torch.Tensor: Reconstructed image blocks with shape (B, num_blocks, 8, 8).
        """

        # Reorder the coefficients back to their original 8x8 block positions using inverse_index
        o = x[..., self.inverse_index]

        # Extract the dimensions after reordering
        B, num_blocks, _ = o.shape

        # Reshape the flat coefficients back into 8x8 blocks
        o = o.view(B, num_blocks, 8, 8)

        return o


@final
class JPEGModel:
    y: torch.Tensor
    cb: torch.Tensor
    cr: torch.Tensor
    y_size: tuple[int, int]
    cb_size: tuple[int, int]
    cr_size: tuple[int, int]

    def __init__(
        self,
        y: torch.Tensor,
        cb: torch.Tensor,
        cr: torch.Tensor,
        y_size: tuple[int, int],
        cb_size: tuple[int, int],
        cr_size: tuple[int, int],
    ) -> None:
        self.y = y
        self.cb = cb
        self.cr = cr
        self.y_size = y_size
        self.cb_size = cb_size
        self.cr_size = cr_size


@final
class JPEGEncoder(nn.Module):
    """
    Module to encode an image into its JPEG compressed representation.

    The encoder performs the following steps:
            1. Converts the input RGB image to YCbCr color space.
            2. Applies chroma subsampling to reduce color information.
            3. Splits the image into non-overlapping blocks.
            4. Applies the Discrete Cosine Transform (DCT) to each block.
            5. Quantizes the DCT coefficients using predefined quantization tables.
            6. rearrange the 8x8 block of DCT coefficients into a 1D array

    Args:
            chroma_downsample_factor (int, optional): Factor by which to downsample chroma channels.
                                                                                               Common factors are 1 (no subsampling), 2, or 4.
                                                                                               Default is 4.
            quality (int, optional): Quality factor for compression (1-100). Higher values preserve
                                                             more details but result in larger file sizes. Default is 100.
    """

    def __init__(
        self, chroma_downsample_factor: Literal[1, 2, 4] = 4, quality: int = 100
    ):
        super().__init__()

        # Convert the quality factor to a scaling factor for quantization tables
        quantization_factor = quality_to_factor(quality)

        # Initialize the color space conversion module (RGB to YCbCr)
        self.color_conversion = ColorConversion()

        # Initialize the chroma subsampling module with the specified factor
        self.chroma_downsample = ChromaDownsample(chroma_downsample_factor)

        # Initialize the block splitting module for dividing the image into blocks
        self.block_splitting = BlockSplitting(3)

        # Initialize the 2D DCT module for transforming blocks to the frequency domain
        self.dct = DCT2D()

        # Initialize quantization modules for luminance (Y) and chrominance (Cb, Cr) channels
        self.quantize_Y = Quantization(QY, quantization_factor)
        self.quantize_C = Quantization(QC, quantization_factor)

        # Initialize zigzag module
        self.zigzag = ZigZagOrder()

    @override
    def forward(self, x: torch.Tensor) -> JPEGModel:
        """
        Encodes the input image into its JPEG compressed representation.

        Args:
                image (torch.Tensor): Input image tensor of shape (B, C, H, W) in RGB format.

        Returns:
                dict: Dictionary containing quantized Y, Cb, Cr blocks and their original sizes.
                          - 'Y': Quantized luminance blocks.
                          - 'Cb': Quantized blue-difference chroma blocks.
                          - 'Cr': Quantized red-difference chroma blocks.
                          - 'Y_size': Original size of the Y channel before block splitting.
                          - 'Cb_size': Original size of the Cb channel before block splitting.
                          - 'Cr_size': Original size of the Cr channel before block splitting.
        """

        # Convert the RGB image to YCbCr color space
        ycbcr = self.color_conversion(x)

        # Apply chroma subsampling to reduce the resolution of Cb and Cr channels
        Y, Cb, Cr = self.chroma_downsample(ycbcr)

        # Split each channel into non-overlapping blocks
        Y_blocks = self.block_splitting(Y)
        Cb_blocks = self.block_splitting(Cb)
        Cr_blocks = self.block_splitting(Cr)

        # Apply the 2D DCT to each block to transform to the frequency domain
        Y_dct = self.dct(Y_blocks)
        Cb_dct = self.dct(Cb_blocks)
        Cr_dct = self.dct(Cr_blocks)

        # Quantize the DCT coefficients using the quantization tables
        Y_quantized = self.quantize_Y(Y_dct)
        Cb_quantized = self.quantize_C(Cb_dct)
        Cr_quantized = self.quantize_C(Cr_dct)

        # Apply zig zag ordering
        Y_zigzag: torch.Tensor = self.zigzag(Y_quantized)
        Cb_zigzag: torch.Tensor = self.zigzag(Cb_quantized)
        Cr_zigzag: torch.Tensor = self.zigzag(Cr_quantized)

        # Return the quantized coefficients along with their original sizes for decoding
        return JPEGModel(
            y=Y_zigzag,
            cb=Cb_zigzag,
            cr=Cr_zigzag,
            y_size=(Y.shape[2], Y.shape[3]),
            cb_size=(Cb.shape[2], Cb.shape[3]),
            cr_size=(Cr.shape[2], Cr.shape[3]),
        )


@final
class JPEGDecoder(nn.Module):
    """
    Module to decode a JPEG compressed representation back into an image.

    The decoder performs the following steps:
            1. Invert the zig zag order.
            2. Dequantizes the quantized DCT coefficients using predefined quantization tables.
            3. Applies the Inverse Discrete Cosine Transform (IDCT) to each block.
            4. Merges the blocks back into full-sized channels.
            5. Upsamples the chroma channels to restore original color information.
            6. Converts the image from YCbCr back to RGB color space.

    Args:
            chroma_upsample_factor (int, optional): Factor by which to upsample chroma channels.
                                                                                             Should match the downsampling factor used during encoding.
                                                                                             Default is 4.
            quality (int, optional): Quality factor for decompression (1-100). Should match the factor used during encoding.
                                                             Default is 100.
    """

    def __init__(
        self, chroma_upsample_factor: Literal[1, 2, 4] = 4, quality: int = 100
    ):
        super().__init__()

        # Invert the zig zag order
        self.inverse_zigzag = InverseZigZagOrder()

        # Convert the quality factor to a scaling factor for dequantization tables
        quantization_factor = quality_to_factor(quality)

        # Initialize dequantization modules for luminance (Y) and chrominance (Cb, Cr) channels
        self.dequantize_Y = Dequantization(QY, quantization_factor)
        self.dequantize_C = Dequantization(QC, quantization_factor)

        # Initialize the 2D IDCT module for transforming frequency domain blocks back to spatial domain
        self.idct = IDCT2D()

        # Initialize the block merging module for reconstructing full-sized channels from blocks
        self.block_merging = BlockMerging(3)

        # Initialize the chroma upsampling module to restore the resolution of Cb and Cr channels
        self.chroma_upsampling = ChromaUpsample(chroma_upsample_factor)

        # Initialize the color space conversion module (YCbCr to RGB)
        self.invert_color_conversion = InvertColorConversion()

    @override
    def forward(self, data: JPEGModel) -> torch.Tensor:
        """
        Decodes the JPEG compressed data back into an RGB image.

        Args:
                data (JPEGModel): Dictionary containing quantized Y, Cb, Cr blocks and their original sizes.
                                         - y: Quantized luminance blocks.
                                         - cb: Quantized blue-difference chroma blocks.
                                         - cr: Quantized red-difference chroma blocks.
                                         - y_size: Original size of the Y channel before block splitting.
                                         - cb_size: Original size of the Cb channel before block splitting.
                                         - cr_size: Original size of the Cr channel before block splitting.

        Returns:
                torch.Tensor: Reconstructed RGB image tensor of shape (B, C, H, W).
        """
        # Extract quantized DCT coefficients and their original sizes from the input data
        Y_zigzag = data.y
        Cb_zigzag = data.cb
        Cr_zigzag = data.cr
        Y_size = data.y_size
        Cb_size = data.cb_size
        Cr_size = data.cr_size

        # Apply reverse zig zag
        Y_quantized = self.inverse_zigzag(Y_zigzag)
        Cb_quantized = self.inverse_zigzag(Cb_zigzag)
        Cr_quantized = self.inverse_zigzag(Cr_zigzag)

        # Dequantize the DCT coefficients to restore their original scale
        Y_dct = self.dequantize_Y(Y_quantized)
        Cb_dct = self.dequantize_C(Cb_quantized)
        Cr_dct = self.dequantize_C(Cr_quantized)

        # Apply the Inverse DCT to each block to transform back to the spatial domain
        Y_blocks = self.idct(Y_dct)
        Cb_blocks = self.idct(Cb_dct)
        Cr_blocks = self.idct(Cr_dct)

        # Merge the blocks back into full-sized channels using their original sizes
        Y = self.block_merging(Y_blocks, Y_size)
        Cb = self.block_merging(Cb_blocks, Cb_size)
        Cr = self.block_merging(Cr_blocks, Cr_size)

        # Upsample the chroma channels to restore their original resolution
        Cb, Cr = self.chroma_upsampling(Cb, Cr)

        # Concatenate the Y, Cb, and Cr channels along the channel dimension
        ycbcr = torch.cat([Y, Cb, Cr], dim=1)

        # Convert the image from YCbCr back to RGB color space
        image = self.invert_color_conversion(ycbcr)

        return image


@final
class JPEG(nn.Module):
    def __init__(
        self, sample_factor: Literal[1, 2, 4] = 4, quality: int | tuple[int, int] = 100
    ):
        super(JPEG, self).__init__()

        if isinstance(quality, tuple):
            quality = random.randint(quality[0], quality[1] + 1)

        self.jpeg_encoder = JPEGEncoder(
            chroma_downsample_factor=sample_factor, quality=quality
        )
        self.jpeg_decoder = JPEGDecoder(
            chroma_upsample_factor=sample_factor, quality=quality
        )

    @override
    def forward(self, x: torch.Tensor):
        o = self.jpeg_encoder(x)
        o = self.jpeg_decoder(o)

        return o


@final
class Degrader1(nn.Module):
    def __init__(self, image_size: int, poisson_alpha: float):
        super(Degrader1, self).__init__()

        self.image_size = image_size
        self.poisson_alpha = poisson_alpha

        self.jpeg = JPEG(quality=(30, 95))

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

        # o = o * 255
        # o = o.to(torch.uint8)
        # o = self.jpeg(o) / 255.0
        o = self.jpeg(o)

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

        # o = o * 255.0
        # o = o.to(torch.uint8)
        # o = self.jpeg(o) / 255.0
        o = self.jpeg(o)

        if np.random.rand() < 0.8:
            o = torch.sinc(o)

        return o


@ModelRegistry.register("real_esrgan")
@final
class RealESRGAN(GANBaseModel):
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
