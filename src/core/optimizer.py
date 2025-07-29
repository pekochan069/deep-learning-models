from typing import Optional, TypedDict, Union
import torch
import torch.optim as optim

from core.utils import TypedDictWithDefaultsMeta

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


class AdafactorParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    beta2_decay: float
    eps: tuple[Optional[float], float]
    d: float
    weight_decay: float
    foreach: Optional[bool]
    maximize: bool


class AdadeltaParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    rho: float
    eps: float
    weight_decay: float
    foreach: Optional[bool]
    capturable: bool
    maximize: bool
    differentiable: bool


class AdagradParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    lr_decay: float
    weight_decay: float
    initial_accumulator_value: float
    eps: float
    foreach: Optional[bool]
    maximize: bool
    differentiable: bool
    fused: Optional[bool]


class AdamParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    betas: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]
    eps: float
    weight_decay: float
    amsgrad: bool
    foreach: Optional[bool]
    maximize: bool
    capturable: bool
    differentiable: bool
    fused: Optional[bool]
    decoupled_weight_decay: bool


class AdamaxParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    foreach: Optional[bool]
    maximize: bool
    differentiable: bool
    capturable: bool


class AdamWParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    betas: tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]
    eps: float
    weight_decay: float
    amsgrad: bool
    maximize: bool
    foreach: Optional[bool]
    capturable: bool
    differentiable: bool
    fused: Optional[bool]


class ASGDParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    lambd: float
    alpha: float
    t0: float
    weight_decay: float
    foreach: Optional[bool]
    maximize: bool
    differentiable: bool
    capturable: bool


class LBFGSParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    max_iter: int
    max_eval: Optional[int]
    tolerance_grad: float
    tolerance_change: float
    history_size: int
    line_search_fn: Optional[str]


class NadamParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    momentum_decay: float
    decoupled_weight_decay: bool
    foreach: Optional[bool]
    maximize: bool
    capturable: bool
    differentiable: bool


class RAdamParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    decoupled_weight_decay: bool
    foreach: Optional[bool]
    maximize: bool
    capturable: bool
    differentiable: bool


class RMSpropParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    alpha: float
    eps: float
    weight_decay: float
    momentum: float
    centered: bool
    capturable: bool
    foreach: Optional[bool]
    maximize: bool
    differentiable: bool


class RpropParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    etas: tuple[float, float]
    step_sizes: tuple[float, float]
    capturable: bool
    foreach: Optional[bool]
    maximize: bool
    differentiable: bool


class SGDParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    momentum: float
    dampening: float
    weight_decay: Union[float, torch.Tensor]
    nesterov: bool
    maximize: bool
    foreach: Optional[bool]
    differentiable: bool
    fused: Optional[bool]


class SparseAdamParams(TypedDictWithDefaultsMeta):
    lr: Union[float, torch.Tensor]
    betas: tuple[float, float]
    eps: float
    maximize: bool


OptimizerParams = Union[
    AdafactorParams,
    AdadeltaParams,
    AdagradParams,
    AdamParams,
    AdamaxParams,
    AdamWParams,
    ASGDParams,
    LBFGSParams,
    NadamParams,
    RAdamParams,
    RMSpropParams,
    RpropParams,
    SGDParams,
    SparseAdamParams,
]
