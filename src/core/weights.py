import os
import torch.nn as nn
from safetensors.torch import save_file, load_file


def save_model(model: nn.Module, name: str):
    if not os.path.exists("weights"):
        os.makedirs("weights")

    save_file(
        model.state_dict(),
        f"weights/{name}.safetensors",
        metadata={"format": "pytorch", "name": name},
    )


def load_model(model: nn.Module, name: str) -> nn.Module | None:
    if not os.path.exists(f"weights/{name}.safetensors"):
        return None

    state_dict = load_file(f"weights/{name}.safetensors")
    _ = model.load_state_dict(state_dict)
    return model
