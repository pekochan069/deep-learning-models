from typing import Literal
import torch.optim as optim

optimizer_names = Literal["sgd", "adam", "adamw"]


def get_optimizer(name: optimizer_names) -> type[optim.Optimizer]:
    match name:
        case "sgd":
            return optim.SGD
        case "adam":
            return optim.Adam
        case "adamw":
            return optim.AdamW
