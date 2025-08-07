from typing import final, override

import torch
import torch.nn as nn
import torchvision.models as models

from core.loss_functions.esrgan_discriminator_loss import ESRGANDiscriminatorLoss  # pyright: ignore[reportMissingTypeStubs]


@final
class VGGFeatureExtractor(nn.Module):
    vgg_layers: nn.Sequential

    def __init__(self, layer_name_list: list[str]):
        super(VGGFeatureExtractor, self).__init__()

        # 사전 훈련된 VGG19 모델 로드
        # 중요: weights 파라미터로 사전 훈련된 가중치 사용을 명시
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # 특징 추출만 필요하므로 features 부분만 사용
        self.vgg_layers = vgg19.features  # pyright: ignore[reportAttributeAccessIssue]

        # 가중치 업데이트 방지 - 매우 중요!
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # 평가 모드로 설정 (BatchNorm, Dropout 등의 동작 고정)
        _ = self.vgg_layers.eval()

        # 어떤 층의 특징을 추출할지 정의
        self.layer_name_list = layer_name_list

        # VGG 층 이름 매핑 생성
        self.layer_name_mapping = self._create_layer_mapping()

    def _create_layer_mapping(self):
        """VGG 층 번호와 이름을 매핑하는 딕셔너리 생성"""
        layer_mapping = {}
        layer_names = [
            "conv1_1",
            "relu1_1",
            "conv1_2",
            "relu1_2",
            "pool1",
            "conv2_1",
            "relu2_1",
            "conv2_2",
            "relu2_2",
            "pool2",
            "conv3_1",
            "relu3_1",
            "conv3_2",
            "relu3_2",
            "conv3_3",
            "relu3_3",
            "conv3_4",
            "relu3_4",
            "pool3",
            "conv4_1",
            "relu4_1",
            "conv4_2",
            "relu4_2",
            "conv4_3",
            "relu4_3",
            "conv4_4",
            "relu4_4",
            "pool4",
            "conv5_1",
            "relu5_1",
            "conv5_2",
            "relu5_2",
            "conv5_3",
            "relu5_3",
            "conv5_4",
            "relu5_4",
            "pool5",
        ]

        for idx, name in enumerate(layer_names):
            layer_mapping[name] = idx

        return layer_mapping

    @override
    def forward(self, x: torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std

        features = {}

        for idx, layer in enumerate(self.vgg_layers):
            x = layer(x)

            # 현재 층이 우리가 관심 있는 층인지 확인
            current_layer_names = [
                name
                for name, layer_idx in self.layer_name_mapping.items()
                if layer_idx == idx
            ]

            for layer_name in current_layer_names:
                if layer_name in self.layer_name_list:
                    features[layer_name] = x.clone()  # 복사본 저장

        return features


@final
class ContentLoss(nn.Module):
    def __init__(self, layer_weights: dict[str, float] | None = None) -> None:
        super(ContentLoss, self).__init__()

        self.feature_extractor = VGGFeatureExtractor(layer_name_list=["conv5_4"])
        self.l1_loss = nn.L1Loss()

        if layer_weights is None:
            self.layer_weights = {"conv5_4": 1.0}
        else:
            self.layer_weights = layer_weights

    @override
    def forward(self, x: torch.Tensor, g_z: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        x_features = self.feature_extractor(x)
        g_z_features = self.feature_extractor(g_z)

        total_loss = torch.tensor(0.0, device=x.device)

        # 각 층별로 손실 계산
        for layer_name in self.layer_weights.keys():
            if layer_name in g_z_features and layer_name in x_features:
                # 특징 맵 크기 정보
                x_feature = x_features[layer_name]
                g_z_feature = g_z_features[layer_name]

                # 정규화를 위한 크기 계산
                # _, channels, height, width = g_z_feature.size()

                # L1 계산 및 정규화
                layer_loss = self.l1_loss(g_z_feature, x_feature)

                # 논문의 정규화 방식 적용: feature map 크기로 나누기
                # normalized_loss = layer_loss / (channels * height * width)

                # 가중치 적용하여 총 손실에 추가
                total_loss += self.layer_weights[layer_name] * layer_loss

        return total_loss


@final
class ESRGANGeneratorLoss(nn.Module):
    def __init__(
        self,
        fake_label: float,
        real_label: float,
    ) -> None:
        super(ESRGANGeneratorLoss, self).__init__()

        self.content_loss = ContentLoss()
        self.adversarial_loss = ESRGANDiscriminatorLoss(
            fake_label=fake_label, real_label=real_label
        )
        self.pointwise_loss = nn.L1Loss()

    @override
    def forward(
        self, x: torch.Tensor, g_z: torch.Tensor, d_g_z: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.content_loss(x, g_z)
            + 5e-3 * self.adversarial_loss(d_g_z, torch.ones_like(d_g_z))
            + 1e-2 * self.pointwise_loss(g_z, x)
        )
