import torch


def get_device():
    return torch.device(available_device())


def available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"
