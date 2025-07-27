import torch.optim as optim

from .names import optimizer_names


def get_optimizer(name: optimizer_names) -> type[optim.Optimizer]:
    match name:
        case "adafactor":
            return optim.Adafactor
        case "adadelta":
            return optim.Adadelta
        case "adagrad":
            return optim.Adagrad
        case "adam":
            return optim.Adam
        case "adamax":
            return optim.Adamax
        case "adamw":
            return optim.AdamW
        case "asgd":
            return optim.ASGD
        case "lbfgs":
            return optim.LBFGS
        case "nadam":
            return optim.NAdam
        case "radam":
            return optim.RAdam
        case "rmsprop":
            return optim.RMSprop
        case "rprop":
            return optim.Rprop
        case "sgd":
            return optim.SGD
        case "sparse_adam":
            return optim.SparseAdam
