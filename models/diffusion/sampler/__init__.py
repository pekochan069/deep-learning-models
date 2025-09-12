from typing import Callable, Protocol, Self, final

import torch
import torch.nn as nn

from models.diffusion.names import SamplerName, SchedulerName
from models.diffusion.scheduler.inverse_quadratic import inverse_quadratic_scheduler
from models.diffusion.scheduler.linear import linear_scheduler
from models.diffusion.scheduler.quadratic import quadratic_scheduler

from .ddpm import DDPMSampler
from .ddim import DDIMSampler
from ..scheduler import (
    InverseQuadraticScheduler,
    LinearScheduler,
    QuadraticScheduler,
    SchedulerProtocol,
)


class SamplerProtocol(Protocol):
    def train_step(
        self,
        x: torch.Tensor,
        sqrt_alphas_bar: torch.Tensor,
        sqrt_one_minus_alphas_bar: torch.Tensor,
    ) -> torch.Tensor: ...

    def step(
        self,
        o: torch.Tensor,
        x_t: torch.Tensor,
        z: torch.Tensor,
        betas: torch.Tensor,
        sqrt_alphas: torch.Tensor,
        sqrt_one_minus_alphas_bar: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor: ...


@final
class Sampler(nn.Module):
    scheduler_name: SchedulerName
    sampler_name: SamplerName
    sampler: SamplerProtocol

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_bar: torch.Tensor
    sqrt_alphas: torch.Tensor
    sqrt_alphas_bar: torch.Tensor
    sqrt_one_minus_alphas_bar: torch.Tensor
    alphas_bar_t_minus_one: torch.Tensor
    sigma_square: torch.Tensor
    sigma: torch.Tensor
    alphas_t_minus_one: torch.Tensor
    sqrt_alphas_t_minus_one: torch.Tensor

    train_step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    step_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]

    def __init__(
        self,
        scheduler: SchedulerName,
        sampler: SamplerName,
        device: torch.device,
        max_T: int = 1000,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()

        self.scheduler_name = scheduler
        self.sampler_name = sampler

        self.max_T = max_T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.dtype = dtype
        self.device = device

        self.set_scheduler()

        match sampler:
            case "ddpm":
                self.train_step_fn = self.ddpm_train
                self.step_fn = self.ddpm
            case "ddim":
                self.train_step_fn = self.ddim_train
                self.step_fn = self.ddim

        # match scheduler:
        # 	case "linear":
        # 		self.scheduler = LinearScheduler(max_T, beta_1, beta_T, dtype, device)
        # 	case "quadratic":
        # 		self.scheduler = QuadraticScheduler(
        # 			max_T, beta_1, beta_T, dtype, device
        # 		)
        # 	case "inverse_quadratic":
        # 		self.scheduler = InverseQuadraticScheduler(
        # 			max_T, beta_1, beta_T, dtype, device
        # 		)

        match sampler:
            case "ddpm":
                self.sampler = DDPMSampler()
            case "ddim":
                self.sampler = DDIMSampler()

    def train_step(self, x: torch.Tensor, t: torch.Tensor):
        # c1, c2 = self.scheduler.train_step(t)
        # x_t = self.sampler.train_step(x, c1, c2)

        # return x_t
        return self.train_step_fn(x, t)

    def step(
        self,
        o: torch.Tensor,
        x_t: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ):
        # c1, c2, c3, c4 = self.scheduler.step(t)
        # x_t = self.sampler.step(o, x_t, z, c1, c2, c3, c4)

        # return x_t
        return self.step_fn(o, x_t, z, t)

    def set_scheduler(self):
        match self.scheduler_name:
            case "linear":
                betas, alphas = linear_scheduler(
                    self.max_T, self.beta_1, self.beta_T, self.dtype, self.device
                )
            case "quadratic":
                betas, alphas = quadratic_scheduler(
                    self.max_T, self.beta_1, self.beta_T, self.dtype, self.device
                )
            case "inverse_quadratic":
                betas, alphas = inverse_quadratic_scheduler(
                    self.max_T, self.beta_1, self.beta_T, self.dtype, self.device
                )

        self.betas = betas
        self.alphas = alphas
        self.sqrt_alphas = self.alphas.sqrt()
        self.alphas_bar = torch.cumprod(self.alphas, -1)
        # self.alphas_bar = self.alphas_bar.clamp(1e-12, 1) # 안정화
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)

        self.alphas_t_minus_one = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alphas[:-1]], dim=0
        )
        self.sqrt_alphas_t_minus_one = self.alphas_t_minus_one.sqrt()

        # posterior variance
        self.alphas_bar_t_minus_one = torch.cat(
            [
                torch.tensor([1.0], device=self.device),
                self.alphas_bar[:-1],
            ],
            dim=0,
        )
        self.sigma_square = (
            (1 - self.alphas_bar_t_minus_one) / (1 - self.alphas_bar) * self.betas
        )
        self.sigma = self.sigma_square.sqrt()

    def ddpm_train(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_bar = self.sqrt_alphas_bar.index_select(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.index_select(
            0, t
        ).view(-1, 1, 1, 1)

        x_t = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar

        return x_t

    def ddpm(
        self, o: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        betas = self.betas.index_select(0, t).view(-1, 1, 1, 1)
        sqrt_alphas = self.sqrt_alphas.index_select(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.index_select(
            0, t
        ).view(-1, 1, 1, 1)
        sigma = self.sigma.index_select(0, t).view(-1, 1, 1, 1)

        c1 = 1 / sqrt_alphas
        c2 = c1 * (betas / sqrt_one_minus_alphas_bar)

        x_t = c1 * x_t - c2 * o + sigma * z

        return x_t

    def ddim_train(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_bar = self.sqrt_alphas_bar.index_select(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar.index_select(
            0, t
        ).view(-1, 1, 1, 1)

        x_t = sqrt_alphas_bar * x + sqrt_one_minus_alphas_bar

        return x_t

    def ddim(
        self, o: torch.Tensor, x_t: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        betas = self.betas.index_select(0, t).view(-1, 1, 1, 1)
        sqrt_alphas = self.sqrt_alphas.index_select(0, t).view(-1, 1, 1, 1)
        alphas_t_minus_one = self.alphas_t_minus_one.index_select(0, t).view(
            -1, 1, 1, 1
        )
        sqrt_alphas_t_minus_one = self.sqrt_alphas_t_minus_one.index_select(0, t).view(
            -1, 1, 1, 1
        )
        sigma_square = self.sigma_square.index_select(0, t).view(-1, 1, 1, 1)
        sigma = self.sigma.index_select(0, t).view(-1, 1, 1, 1)

        c1 = sqrt_alphas_t_minus_one / sqrt_alphas
        c2_1 = (1 - alphas_t_minus_one - sigma_square).sqrt()
        c2_2 = sqrt_alphas_t_minus_one * betas.sqrt() / sqrt_alphas
        c2 = c2_1 - c2_2

        return c1 * x_t - c2 * o + sigma * z
