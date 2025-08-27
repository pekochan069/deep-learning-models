from core.config import DiffusionConfig
from .vae import SimpleVAE, CVAE, ConditionalVAE


def get_diffusion_model(config: DiffusionConfig):
    match config.model:
        case "simple_vae":
            return SimpleVAE(config, **config.model_params.to_kwargs())
        case "cvae":
            return CVAE(config, **config.model_params.to_kwargs())
        case "conditional_vae":
            return ConditionalVAE(config, **config.model_params.to_kwargs())


__all__ = ["get_diffusion_model"]
