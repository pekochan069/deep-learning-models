from typing import final

import torch


def linear_scheduler(
    max_T: int,
    beta_1: float,
    beta_T: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(beta_1, beta_T, max_T, dtype=dtype, device=device)
    # self.betas = self.betas.clamp(1e-8, 0.999) # 안정화
    alphas: torch.Tensor = 1 - betas

    return betas, alphas


@final
class LinearScheduler:
    def __init__(
        self,
        max_T: int,
        beta_1: float,
        beta_T: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.betas = torch.linspace(beta_1, beta_T, max_T, dtype=dtype, device=device)
        # self.betas = self.betas.clamp(1e-8, 0.999) # 안정화
        self.alphas: torch.Tensor = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, -1)
        # self.alphas_bar = self.alphas_bar.clamp(1e-12, 1) # 안정화
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)

        # posterior variance
        self.alphas_bar_t_minus_1 = torch.cat(
            [
                torch.tensor([1.0], device=device),
                self.alphas_bar[:-1],
            ],
            dim=0,
        )
        self.sigma: torch.Tensor = (
            (1 - self.alphas_bar_t_minus_1) / (1 - self.alphas_bar) * self.betas
        ).sqrt()
        # self.posterior_betas  torch.log(self.posterior_betas.clamp(min=1e-20))

    def train_step(self, t: torch.Tensor):
        with torch.no_grad():
            sqrt_alphas_bar = self.sqrt_alphas_bar.index_select(0, t).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.index_select(
                0, t
            ).view(-1, 1, 1, 1)

        return sqrt_alphas_bar, sqrt_one_minus_alphas_bar

    def step(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            betas = self.betas.index_select(0, t).view(-1, 1, 1, 1)
            sqrt_alphas = self.alphas.sqrt().index_select(0, t).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.index_select(
                0, t
            ).view(-1, 1, 1, 1)
            sigma = self.sigma.index_select(0, t).view(-1, 1, 1, 1)

        return betas, sqrt_alphas, sqrt_one_minus_alphas_bar, sigma
