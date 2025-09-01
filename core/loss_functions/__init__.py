from typing import Annotated
from pydantic import Field

from .gan_discriminator_loss import GANDiscriminatorLoss, GANDiscriminatorLossParams
from .srgan_generator_loss import SRGANGeneratorLoss, SRGANGeneratorLossParams
from .esrgan_generator_loss import ESRGANGeneratorLoss, ESRGANGeneratorLossParams
from .esrgan_discriminator_loss import (
    ESRGANDiscriminatorLoss,
    ESRGANDiscriminatorLossParams,
)
from .vae_loss import VAELoss, VAELossParams
from .beta_vae_loss import BetaVAELoss, BetaVAELossParams
from .vq_vae_loss import VQVAELoss, VQVAELossParams

LossParams = Annotated[
    GANDiscriminatorLossParams
    | SRGANGeneratorLossParams
    | ESRGANGeneratorLossParams
    | ESRGANDiscriminatorLossParams
    | VAELossParams
    | BetaVAELossParams
    | VQVAELossParams,
    Field(discriminator="param_type"),
]

__all__ = [
    "LossParams",
    "GANDiscriminatorLoss",
    "GANDiscriminatorLossParams",
    "SRGANGeneratorLoss",
    "SRGANGeneratorLossParams",
    "ESRGANGeneratorLoss",
    "ESRGANGeneratorLossParams",
    "ESRGANDiscriminatorLoss",
    "ESRGANDiscriminatorLossParams",
    "VAELoss",
    "VAELossParams",
    "BetaVAELoss",
    "BetaVAELossParams",
    "VQVAELoss",
    "VQVAELossParams",
]
