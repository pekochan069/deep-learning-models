import os
import tempfile
import time
from typing import Any
from warnings import deprecated

import torch
import torch.nn as nn
from pydantic import BaseModel
from safetensors.torch import save_file, load_file


class TrainerState(BaseModel):
    epoch: int | None
    best_metric: float | None


def atomic_write(write_fn: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as f:
        tmp = f.name
        write_fn(tmp)
    os.replace(tmp, path)


def model_dir(name: str, label: str = "last") -> str:
    return os.path.join("weights", name, label)


def model_path(name: str, label: str = "last") -> str:
    return os.path.join(model_dir(name, label), "model.safetensors")


def ema_path(name: str, label: str = "last") -> str:
    return os.path.join(model_dir(name, label), "ema.safetensors")


def state_json_path(name: str, label: str = "last") -> str:
    return os.path.join(model_dir(name, label), "trainer_state.json")


def optim_blob_path(name: str, label: str = "last") -> str:
    return os.path.join(model_dir(name, label), "optim.pt")


def save_model(
    model: nn.Module,
    name: str,
    label: str,
    *,
    epoch: int | None = None,
    best_metric: float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: nn.Module | None = None,
    scaler: nn.Module | None = None,
    ema_state: dict[Any, Any] | None = None,
    meta: dict[str, Any] | None = None,
):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    metadata = {
        "format": "pytorch",
        "torch": torch.__version__,
        "name": name,
        "tag": label,
        "time": str(time.time()),
        "params": str(sum(p.numel() for p in model.parameters())),
        **{k: str(v) for k, v in (meta or {}).items()},
    }

    atomic_write(
        lambda p: save_file(state, p, metadata=metadata), model_path(name, label)
    )

    if ema_state is not None:
        ema_state = {k: v.detach().cpu() for k, v in ema_state.items()}
        atomic_write(
            lambda p: save_file(
                ema_state, p, metadata={"name": name, "label": label, "ema": "true"}
            ),
            ema_path(name, label),
        )

    trainer_state = TrainerState(epoch=epoch, best_metric=best_metric)
    atomic_write(
        lambda p: open(p, "w", encoding="utf-8").write(
            trainer_state.model_dump_json(indent=2)
        ),
        state_json_path(name, label),
    )

    if optimizer is not None or lr_scheduler is not None or scaler is not None:
        blob = {
            "optimizer": optimizer.state_dict() if optimizer else None,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
        }
        torch.save(blob, optim_blob_path(name, label))


def load_model(model: nn.Module, name: str, label: str, strict: bool = True):
    # state_dict = load_file(f"weights/{name}/{label}.safetensors")
    # _ = model.load_state_dict(state_dict)
    # return model
    path = model_path(name, label)

    if not os.path.exists(path):
        return None

    loaded = load_file(path)
    missing, unexpected = model.load_state_dict(loaded, strict=strict)
    if missing or unexpected:
        print(f"[load_model:{name}/{label}] missing={missing} unexpected={unexpected}")
    return model


def load_ema_state(name: str, label: str = "last"):
    path = ema_path(name, label)
    if not os.path.exists(path):
        return None
    return load_file(path)


@deprecated("Use save_model instead")
def save_model_old(model: nn.Module, name: str):
    """deprecated"""
    if not os.path.exists("weights"):
        os.makedirs("weights")

    save_file(
        model.state_dict(),
        f"weights/{name}.safetensors",
        metadata={"format": "pytorch", "name": name},
    )


@deprecated("Use load_model instead")
def load_model_old(model: nn.Module, name: str) -> nn.Module | None:
    """deprecated"""
    if not os.path.exists(f"weights/{name}.safetensors"):
        return None

    state_dict = load_file(f"weights/{name}.safetensors")
    _ = model.load_state_dict(state_dict)
    return model


__all__ = [
    "save_model",
    "load_model",
    "load_ema_state",
    "save_model_old",
    "load_model_old",
]
