import torch


def set_seed(seed: int | None = None):
    if seed is not None:
        _ = torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
