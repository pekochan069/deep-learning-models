from typing import Any

import torch.nn as nn

from core.loss_functions.vae_loss import VAELoss

from .names import LossFunctionName
from .loss_functions.gan_discriminator_loss import GANDiscriminatorLoss
from .loss_functions.srgan_generator_loss import SRGANGeneratorLoss
from .loss_functions.esrgan_discriminator_loss import ESRGANDiscriminatorLoss


def get_loss_function(name: LossFunctionName, params: dict[str, Any] | None = None):
    if params is None:
        params = {}

    match name:
        case "l1":
            return nn.L1Loss(**params)
        case "nll":
            return nn.NLLLoss(**params)
        case "poisson_nll":
            return nn.PoissonNLLLoss(**params)
        case "gaussian_nll":
            return nn.GaussianNLLLoss(**params)
        case "kl_div":
            return nn.KLDivLoss(**params)
        case "mse":
            return nn.MSELoss(**params)
        case "bce":
            return nn.BCELoss(**params)
        case "bce_with_logits":
            return nn.BCEWithLogitsLoss(**params)
        case "hinge_embedding":
            return nn.HingeEmbeddingLoss(**params)
        case "multi_label_margin":
            return nn.MultiLabelMarginLoss(**params)
        case "smooth_l1":
            return nn.SmoothL1Loss(**params)
        case "huber":
            return nn.HuberLoss(**params)
        case "soft_margin":
            return nn.SoftMarginLoss(**params)
        case "cross_entropy":
            return nn.CrossEntropyLoss(**params)
        case "multi_label_soft_margin":
            return nn.MultiLabelSoftMarginLoss(**params)
        case "cosine_embedding":
            return nn.CosineEmbeddingLoss(**params)
        case "margin_ranking":
            return nn.MarginRankingLoss(**params)
        case "multi_margin":
            return nn.MultiMarginLoss(**params)
        case "triplet_margin":
            return nn.TripletMarginLoss(**params)
        case "triplet_margin_with_distance":
            return nn.TripletMarginWithDistanceLoss(**params)
        case "ctc":
            return nn.CTCLoss(**params)
        case "gan_discriminator_loss":
            return GANDiscriminatorLoss(**params)
        case "srgan_generator_loss":
            return SRGANGeneratorLoss(**params)
        case "esrgan_discriminator_loss":
            return ESRGANDiscriminatorLoss(**params)
        case "vae_loss":
            return VAELoss(**params)
