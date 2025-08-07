import torch


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "xpu"
        if torch.xpu.is_available()
        else "cpu"
    )
