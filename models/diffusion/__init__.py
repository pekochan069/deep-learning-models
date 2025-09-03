from core.config import DiffusionConfig
from .vae import SimpleVAE, CVAE, ConditionalCVAE, CFGCVAE, CFGVQVAE
from .diffusion import DDPM


def get_diffusion_model(config: DiffusionConfig):
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
        case "ddpm":
            return DDPM(config, **config.model_params.to_kwargs())


__all__ = ["get_diffusion_model"]
