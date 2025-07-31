from typing import Annotated, Literal
import torch
import torch.optim as optim
from pydantic import ConfigDict, Field

from .pydantic import ParametersBase
from .names import OptimizerName


def get_optimizer(name: OptimizerName):
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


class AdafactorParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["adafactor"] = "adafactor"
    lr: float | torch.Tensor = 1e-2
    beta2_decay: float = -0.8
    eps: tuple[float | None, float] = (None, 1e-3)
    d: float = 1.0
    weight_decay: float = 0.0
    foreach: bool | None = None
    maximize: bool = False


class AdadeltaParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["adadelta"] = "adadelta"
    lr: float | torch.Tensor = 1.0
    rho: float = 0.9
    eps: float = 1e-6
    weight_decay: float = 0
    foreach: bool | None = None
    capturable: bool = False
    maximize: bool = False
    differentiable: bool = False


class AdagradParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["adagrad"] = "adagrad"
    r: float | torch.Tensor = 1e-2
    lr_decay: float = 0
    weight_decay: float = 0
    initial_accumulator_value: float = 0
    eps: float = 1e-10
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False
    fused: bool | None = None


class AdamParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["adam"] = "adam"
    lr: float | torch.Tensor = 1e-3
    betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False
    foreach: bool | None = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None
    decoupled_weight_decay: bool = False


class AdamaxParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["adamax"] = "adamax"
    lr: float | torch.Tensor = 2e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False
    capturable: bool = False


class AdamWParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["adamw"] = "adamw"
    lr: float | torch.Tensor = 1e-3
    betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool | None = None
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None


class ASGDParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["asgd"] = "asgd"
    lr: float | torch.Tensor = 1e-2
    lambd: float = 1e-4
    alpha: float = 0.75
    t0: float = 1e6
    weight_decay: float = 0
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False
    capturable: bool = False


class LBFGSParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["lbfgs"] = "lbfgs"
    lr: float | torch.Tensor = 1
    max_iter: int = 20
    max_eval: int | None = None
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str | None = None


class NAdamParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["nadam"] = "nadam"
    lr: float | torch.Tensor = 2e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    momentum_decay: float = 4e-3
    decoupled_weight_decay: bool = False
    foreach: bool | None = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False


class RAdamParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["radam"] = "radam"
    lr: float | torch.Tensor = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    decoupled_weight_decay: bool = False
    foreach: bool | None = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False


class RMSpropParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["rmsprop"] = "rmsprop"
    lr: float | torch.Tensor = 1e-2
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False
    capturable: bool = False
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False


class RpropParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["rprop"] = "rprop"
    lr: float | torch.Tensor = 1e-2
    etas: tuple[float, float] = (0.5, 1.2)
    step_sizes: tuple[float, float] = (1e-6, 50)
    capturable: bool = False
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False


class SGDParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["sgd"] = "sgd"
    lr: float | torch.Tensor = 1e-3
    momentum: float = 0
    dampening: float = 0
    weight_decay: float | torch.Tensor = 0
    nesterov: bool = False
    maximize: bool = False
    foreach: bool | None = None
    differentiable: bool = False
    fused: bool | None = None


class SparseAdamParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _tag: Literal["sparse_adam"] = "sparse_adam"
    lr: float | torch.Tensor = 1e-3
    betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.999)
    eps: float = 1e-8
    maximize: bool = False


OptimizerParams = Annotated[
    AdafactorParams
    | AdadeltaParams
    | AdagradParams
    | AdamParams
    | AdamaxParams
    | AdamWParams
    | ASGDParams
    | LBFGSParams
    | NAdamParams
    | RAdamParams
    | RMSpropParams
    | RpropParams
    | SGDParams
    | SparseAdamParams,
    Field(discriminator="_tag"),
]
