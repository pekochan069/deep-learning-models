from core.config import DiffusionConfig
from .vae import VAE
from .ddpm import DDPM


def get_diffusion_model(config: DiffusionConfig):
    match config.model:
        case "vae":
            return VAE(config)
        case "ddpm":
            return DDPM(config)
