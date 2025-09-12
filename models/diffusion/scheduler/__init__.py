from typing import Protocol

import torch


from .linear import LinearScheduler
from .quadratic import QuadraticScheduler
from .inverse_quadratic import InverseQuadraticScheduler

__all__ = ["LinearScheduler", "QuadraticScheduler", "InverseQuadraticScheduler"]


class SchedulerProtocol(Protocol):
    def train_step(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def step(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
