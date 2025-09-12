from core.config import VAEConfig
from .simple_vae import SimpleVAE
from .cvae import CVAE
from .conditional_cvae import ConditionalCVAE
from .cfg_cvae import CFGCVAE
from .cfg_vq_vae import CFGVQVAE


def get_vae_model(config: VAEConfig):
    match config.model:
        case "simple_vae":
            return SimpleVAE(config, **config.model_params.to_kwargs())
        case "cvae":
            return CVAE(config, **config.model_params.to_kwargs())
        case "conditional_cvae":
            return ConditionalCVAE(config, **config.model_params.to_kwargs())
        case "cfg_cvae":
            return CFGCVAE(config, **config.model_params.to_kwargs())
        case "cfg_vq_vae":
            return CFGVQVAE(config, **config.model_params.to_kwargs())
