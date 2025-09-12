from typing import final

import torch


def inverse_quadratic_scheduler(
    max_T: int,
    beta_1: float,
    beta_T: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(1, max_T + 1, dtype=dtype, device=device)
    t = t / max_T
    # t = torch.linspace(beta_1, beta_T, max_T, device=device)
    betas = beta_1 + (beta_T - beta_1) * (t**2)
    # self.betas = self.betas.clamp(1e-8, 0.999) # 안정화
    alphas: torch.Tensor = 1 - betas
    # alphas_bar = torch.cumprod(alphas, -1)
    # # alphas_bar = alphas_bar.clamp(1e-12, 1) # 안정화
    # sqrt_alphas_bar = torch.sqrt(alphas_bar)
    # sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

    # # posterior variance
    # alphas_bar_t_minus_1 = torch.cat(
    #     [
    #         torch.tensor([1.0], device=device),
    #         alphas_bar[:-1],
    #     ],
    #     dim=0,
    # )
    # sigma_square: torch.Tensor = (1 - alphas_bar_t_minus_1) / (1 - alphas_bar) * betas

    return betas, alphas


@final
class InverseQuadraticScheduler:
    def __init__(
        self,
        max_T: int,
        beta_1: float,
        beta_T: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        t = torch.arange(1, max_T + 1, device=device)
        t = t / max_T
        # t = torch.linspace(beta_1, beta_T, max_T, device=device)
        self.betas = beta_1 + (beta_T - beta_1) * (t**2)
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
