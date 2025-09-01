from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from core.pydantic import ParametersBase


class SimpleVAEParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    param_type: Literal["simple_vae"] = "simple_vae"
    hidden_dim: int = 200
    latent_dim: int = 20


class CVAEParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    param_type: Literal["cvae"] = "cvae"
    hidden_dim: int = 8
    latent_dim: int = 24


class ConditionalCVAEParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    param_type: Literal["conditional_vae"] = "conditional_vae"
    hidden_dim: int = 16
    latent_dim: int = 64


class CFGCVAEParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    param_type: Literal["cfg_cvae"] = "cfg_cvae"
    hidden_dim: int = 16
    latent_dim: int = 64
    embedding_dim: int = 8


class CFGVQVAEParams(ParametersBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    param_type: Literal["cfg_vq_vae"] = "cfg_vq_vae"
    encoder_embedding_dim: int = 8
    vector_dim: int = 512
    embedding_dim: int = 64
    beta: float = 0.25
    gamma: float = 0.99


DiffusionModelParams = Annotated[
    SimpleVAEParams
    | CVAEParams
    | ConditionalCVAEParams
    | CFGCVAEParams
    | CFGVQVAEParams,
    Field(discriminator="param_type"),
]
