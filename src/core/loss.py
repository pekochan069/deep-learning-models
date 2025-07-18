import torch.nn as nn
from typing import Literal

loss_function_names = Literal["cross_entropy", "mse", "nll_loss"]


def get_loss_function(name: loss_function_names) -> nn.Module:
    match name:
        case "cross_entropy":
            return nn.CrossEntropyLoss()
        case "mse":
            return nn.MSELoss()
        case "nll_loss":
            return nn.NLLLoss()
        case _:
            raise ValueError(f"Unknown loss function: {name}")
