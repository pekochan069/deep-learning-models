import torch.optim as optim

from .names import optimizer_names


def get_optimizer(name: optimizer_names) -> type[optim.Optimizer]:
    match name:
        case "sgd":
            return optim.SGD
        case "adam":
            return optim.Adam
        case "adamw":
            return optim.AdamW
