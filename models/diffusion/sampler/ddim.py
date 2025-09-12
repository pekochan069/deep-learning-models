from typing import final
import torch
import torch.nn as nn


@final
class DDIMSampler:
    def __init__(self): ...

    def train_step(
        self,
        x: torch.Tensor,
        sqrt_alphas_bar: torch.Tensor,
        sqrt_one_minus_alphas_bar: torch.Tensor,
    ) -> torch.Tensor:
        x_t = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar

        return x_t

    def step(
        self,
        o: torch.Tensor,
        x_t: torch.Tensor,
        z: torch.Tensor,
        betas: torch.Tensor,
        sqrt_alphas: torch.Tensor,
        sqrt_one_minus_alphas_bar: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        # c1 = 1 / sqrt_alphas
        # c2 = c1 * (betas / sqrt_one_minus_alphas_bar)

        # x_t = c1 * x_t - c2 * o + sigma * z

        return x_t
