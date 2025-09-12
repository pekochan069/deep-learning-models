from core.config import DiffusionConfig
from .ddpm import DDPM


def get_diffusion_model(config: DiffusionConfig):
    match config.model:
        case "ddpm":
            return DDPM(config, **config.model_params.to_kwargs())


__all__ = ["get_diffusion_model"]
