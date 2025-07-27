import torch.nn as nn

from .names import loss_function_names


def get_loss_function(name: loss_function_names) -> nn.Module:
    match name:
        case "l1":
            return nn.L1Loss()
        case "nll":
            return nn.NLLLoss()
        case "poisson_nll":
            return nn.PoissonNLLLoss()
        case "gaussian_nll":
            return nn.GaussianNLLLoss()
        case "kl_div":
            return nn.KLDivLoss()
        case "mse":
            return nn.MSELoss()
        case "bce":
            return nn.BCELoss()
        case "bce_with_logits":
            return nn.BCEWithLogitsLoss()
        case "hinge_embedding":
            return nn.HingeEmbeddingLoss()
        case "multi_label_margin":
            return nn.MultiLabelMarginLoss()
        case "smooth_l1":
            return nn.SmoothL1Loss()
        case "huber":
            return nn.HuberLoss()
        case "soft_margin":
            return nn.SoftMarginLoss()
        case "cross_entropy":
            return nn.CrossEntropyLoss()
        case "multi_label_soft_margin":
            return nn.MultiLabelSoftMarginLoss()
        case "cosine_embedding":
            return nn.CosineEmbeddingLoss()
        case "margin_ranking":
            return nn.MarginRankingLoss()
        case "multi_margin":
            return nn.MultiMarginLoss()
        case "triplet_margin":
            return nn.TripletMarginLoss()
        case "triplet_margin_with_distance":
            return nn.TripletMarginWithDistanceLoss()
        case "ctc":
            return nn.CTCLoss()
