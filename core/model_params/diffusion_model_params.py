from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from core.pydantic import ParametersBase


class DDPMParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    param_type: Literal["ddpm"] = "ddpm"
    max_T: int = 1000
    beta_1: float = 1e-4
    beta_T: float = 0.02
    cfg_unconditional_prob: float = 0.1
    base_unet_dim: int = 64
    t_emb_dim: int = 64
    long_skip_connection_type: Literal["add", "concat"] = "concat"
    dropout: float = 0.1
    skip_reduce_ratio: float = 0.5
    gradient_clipping: bool = True
    max_clip_norm: int = 5


DiffusionModelParams = Annotated[
    DDPMParams,
    Field(discriminator="param_type"),
]
