from core.config import DiffusionConfig
from .vae import SimpleVAE
from .ddpm import DDPM


def get_diffusion_model(config: DiffusionConfig):
    match config.model:
        case "simple_vae":
            return SimpleVAE(config)
        case "ddpm":
            return DDPM(config)
