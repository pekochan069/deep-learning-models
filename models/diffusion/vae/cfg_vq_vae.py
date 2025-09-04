import time
from typing import Literal, final, override

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import DiffusionConfig
from core.dataset import get_channels, get_image_size, get_num_classes
from core.loss import get_loss_function
from core.optimizer import get_optimizer
from .base_vae import VAEBaseModel


@final
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: Literal["A", "B"], *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        mask: torch.Tensor = self.weight.data.clone()
        self.register_buffer("mask", mask)

        _, _, h, w = self.weight.size()
        h: int = h // 2
        w: int = w // 2

        _ = self.mask.fill_(1)  # pyright: ignore[reportCallIssue]
        _ = self.mask[:, :, h, w + 1 :] = 0  # pyright: ignore[reportIndexIssue]
        _ = self.mask[:, :, h + 1 :, :] = 0  # pyright: ignore[reportIndexIssue]

        if mask_type == "A":
            _ = self.mask[:, :, h, w] = 0  # pyright: ignore[reportIndexIssue]

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        masked = self.weight * self.mask  # pyright: ignore[reportOperatorIssue]

        o = F.conv2d(
            input,
            masked,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return o


@final
class GatedBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(GatedBlock, self).__init__()

        self.conv1 = nn.Sequential(
            MaskedConv2d(
                "B",
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_dim),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            MaskedConv2d(
                "B",
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )

        self.conv3 = nn.Conv2d(out_dim, in_dim, 1, 1, 0, bias=False)

    @override
    def forward(self, x: torch.Tensor):
        o1 = self.conv1(x)
        o2 = self.conv2(x)

        o = o1 * o2

        o = self.conv3(o)

        o = x + o

        return x


@final
class PixelCNN(nn.Module):
    def __init__(self, in_dim: int, feature_maps: int, num_classes: int, layers: int):
        super(PixelCNN, self).__init__()

        self.num_classes = num_classes

        layer_list = nn.ModuleList()

        _ = layer_list.extend(
            [
                MaskedConv2d(
                    "A",
                    in_channels=in_dim,
                    out_channels=feature_maps,
                    kernel_size=7,
                    padding=3,
                ),
                nn.BatchNorm2d(feature_maps),
                nn.ReLU(),
            ]
        )

        for _ in range(layers):
            _ = layer_list.append(GatedBlock(feature_maps, feature_maps))

        _ = layer_list.extend(
            [
                nn.ReLU(),
                nn.Conv2d(feature_maps, feature_maps, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(feature_maps, in_dim * num_classes, 1),
            ]
        )

        self.sequential = nn.Sequential(*layer_list)

    @override
    def forward(self, x: torch.Tensor):
        o = self.sequential(x)

        if x.size(1) != 1:
            o = rearrange(o, "b (c classes) h w -> b c classes h w", c=x.size(1))

        return o

    def generate(
        self, n_samples: int, image_size: int, in_dim: int, device: torch.device
    ):
        _ = self.eval()

        samples = torch.zeros(n_samples, in_dim, image_size, image_size, device=device)

        with torch.no_grad():
            for h in tqdm(range(image_size), desc="Generating Image"):
                for w in range(image_size):
                    logits = self(samples)
                    probs = F.softmax(logits[:, :, h, w], dim=1)
                    pixel_sample = torch.multinomial(probs, 1).squeeze(-1)
                    samples[:, 0, h, w] = pixel_sample / (self.num_classes - 1.0)

        return samples


@final
class Encoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder_embedding_dim: int,
        in_dim: int,
        embedding_dim: int,
    ):
        super(Encoder, self).__init__()

        self.in_dim = in_dim

        self.embedding = nn.Embedding(num_classes + 1, encoder_embedding_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_dim + encoder_embedding_dim,
                embedding_dim // 4,
                3,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim // 4),
            nn.ReLU(),
            nn.Conv2d(embedding_dim // 4, embedding_dim // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embedding_dim // 2, embedding_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
        )

    @override
    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.embedding(y)
        y = repeat(y, "b y -> b y h w", h=x.size(2), w=x.size(3))
        x = torch.cat([x, y], dim=1)
        o = self.conv(x)

        return o


@final
class Quantization(nn.Module):
    def __init__(self, vector_dim: int, embedding_dim: int, beta: float):
        super(Quantization, self).__init__()

        self.beta = beta

        self.embedding = nn.Embedding(vector_dim, embedding_dim)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                _ = nn.init.uniform_(m.weight, 1 / vector_dim, 1 / embedding_dim)

    @override
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape
        z = rearrange(x, "b c h w -> (b h w) c")
        e = self.embedding.weight

        e = torch.cdist(z, e)
        e = torch.argmin(e, dim=1)
        indices = e

        e = self.embedding(e)

        e = rearrange(e, "(b h w) c -> b c h w", h=h, w=w)

        codebook_loss = F.mse_loss(x.detach(), e)
        commitment_loss = self.beta * F.mse_loss(x, e.detach())
        loss = codebook_loss + commitment_loss

        o = x + (e - x).detach()

        indices = rearrange(indices, "(b h w) -> b h w", h=h, w=w)

        return o, loss, indices


@final
class Decoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        decoder_embedding_dim: int,
        embedding_dim: int,
        out_dim: int,
    ):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(num_classes + 1, decoder_embedding_dim)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_dim + decoder_embedding_dim,
                embedding_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                embedding_dim // 2,
                embedding_dim // 4,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                embedding_dim // 4,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )

    @override
    def forward(self, q: torch.Tensor, y: torch.Tensor):
        _, _, h, w = q.shape
        y = self.embedding(y)
        y = repeat(y, "b c -> b c h w", h=h, w=w)
        q = torch.cat([q, y], dim=1)
        o = self.deconv(q)

        return o


@final
class CFGVQVAE(VAEBaseModel):
    def __init__(
        self,
        config: DiffusionConfig,
        encoder_embedding_dim: int,
        vector_dim: int,
        embedding_dim: int,
        beta: float,
        gamma: float,
    ):
        super().__init__(config)

        self.image_size = get_image_size(self.config.dataset)
        self.num_classes = get_num_classes(self.config.dataset)
        self.channels = get_channels(self.config.dataset)

        self.vector_dim = vector_dim
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(
            self.num_classes, encoder_embedding_dim, self.channels, self.embedding_dim
        )
        self.quantization = Quantization(self.vector_dim, self.embedding_dim, beta)
        self.decoder = Decoder(
            self.num_classes, encoder_embedding_dim, self.embedding_dim, self.channels
        )
        self.pixel_cnn = PixelCNN(
            in_dim=self.embedding_dim,
            feature_maps=64,
            num_classes=self.vector_dim,
            layers=7,
        )

    def apply_cfg_conditioning(
        self, y: torch.Tensor, uncond_prob: float, uncond_class: int
    ) -> torch.Tensor:
        probs = torch.rand(y.shape[0], device=self.device)
        mask = probs < uncond_prob
        new_y = y.clone()
        new_y[mask] = uncond_class

        return new_y

    @override
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ):
        _ = self.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            x, y = batch

            x = x.to(self.device)
            y = y.to(self.device)

            y = self.apply_cfg_conditioning(y, 0.1, self.num_classes)

            optimizer.zero_grad()

            z = self.encoder(x, y)
            q, q_loss, _ = self.quantization(z)
            x_hat = self.decoder(q, y)

            loss = loss_function(x, x_hat, q_loss)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    @override
    def validate_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        loss_function: nn.Module,
    ):
        _ = self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                z = self.encoder(x, y)
                q, q_loss, _ = self.quantization(z)
                x_hat = self.decoder(q, y)

                loss = loss_function(x, x_hat, q_loss)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    @override
    def predict(self, batch_size: int = 64, guidance_scale: float = 2.0):
        _ = self.eval()

        with torch.no_grad():
            indices = torch.randint(
                0,
                self.embedding_dim,
                (
                    batch_size,
                    (self.image_size // 8) ** 2,
                ),
                device=self.device,
            )
            q = self.quantization.embedding(indices)
            q = rearrange(
                q,
                "b (h w) c -> b c h w",
                h=self.image_size // 8,
                w=self.image_size // 8,
            )
            y = torch.randint(
                0, self.num_classes + 1, (batch_size,), device=self.device
            )
            y_uncond = self.apply_cfg_conditioning(y, 0.1, self.num_classes)
            o = self.decoder(q, y)
            o_uncond = self.decoder(q, y_uncond)
            o = o_uncond + guidance_scale * (o - o_uncond)
            o = o.clamp(0, 1)
            o = o.cpu()

        grid_img = torchvision.utils.make_grid(o, nrow=8, padding=8, normalize=True)
        _ = plt.imshow(grid_img.permute(1, 2, 0))
        _ = plt.axis("off")
        _ = plt.show()

    def train_pixel_cnn_epoch(self):
        _ = self.eval()

        for param in self.parameters():
            param.requires_grad = False

        # TODO Finish PixelCNN Train loop

    @override
    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        self.logger.info(
            f"Training {self.config.model} on {self.config.dataset} dataset"
        )
        start = time.time()  # noqa: F821

        optimizer = get_optimizer(self.config.optimizer)(
            self.parameters(), **self.config.optimizer_params.to_kwargs()
        )
        loss_function = get_loss_function(self.config.loss_function).to(self.device)

        early_stop_counter = 0
        warning_printed = False

        for epoch in range(self.config.epochs):
            self.logger.info(f"Training epoch {epoch + 1}/{self.config.epochs}")

            epoch_loss = self.train_epoch(train_loader, optimizer, loss_function)

            self.history.train_loss.append(epoch_loss)

            if val_loader:
                epoch_val_loss = self.validate_epoch(val_loader, loss_function)
                self.history.val_loss.append(epoch_val_loss)
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} finished, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} finished, Loss: {epoch_loss:.4f}"
                )

            if (
                self.config.save_after_n_epoch
                and (epoch + 1) % self.config.save_after_n_epoch_period == 0
            ):
                self.save(f"{self.config.name}_epoch_{epoch + 1}")
                self.logger.info(f"Model saved at epoch {epoch + 1}")

            if self.config.early_stopping:
                if self.config.early_stopping_monitor == "val_loss":
                    if not val_loader and not warning_printed:
                        warning_printed = True
                        self.logger.warning(
                            "Early stopping is enabled but validation data loader is not provided."
                        )
                        continue

                    if len(self.history.val_loss) < 2:
                        continue

                    if self.config.early_stopping_min_delta_strategy == "fixed":
                        if (
                            self.history.val_loss[-1]
                            < self.history.val_loss[-2]
                            + self.config.early_stopping_min_delta
                        ):
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                    elif (
                        self.config.early_stopping_min_delta_strategy
                        == "previous_proportional"
                    ):
                        if (
                            self.history.val_loss[-1]
                            < self.history.val_loss[-2]
                            + self.history.val_loss[-2]
                            * self.config.early_stopping_min_delta
                        ):
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                    elif (
                        self.config.early_stopping_min_delta_strategy
                        == "delta_proportional"
                    ):
                        if (
                            self.history.val_loss[-1]
                            < self.history.val_loss[-2]
                            + (self.history.val_loss[-3] - self.history.val_loss[-2])
                            * self.config.early_stopping_min_delta
                        ):
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                else:
                    if len(self.history.train_loss) < 2:
                        continue

                    if self.config.early_stopping_min_delta_strategy == "fixed":
                        if (
                            self.history.train_loss[-1]
                            < self.history.train_loss[-2]
                            + self.config.early_stopping_min_delta
                        ):
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                    elif (
                        self.config.early_stopping_min_delta_strategy
                        == "previous_proportional"
                    ):
                        if (
                            self.history.train_loss[-1]
                            < self.history.train_loss[-2]
                            + self.history.train_loss[-2]
                            * self.config.early_stopping_min_delta
                        ):
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                    elif (
                        self.config.early_stopping_min_delta_strategy
                        == "delta_proportional"
                    ):
                        if (
                            self.history.train_loss[-1]
                            < self.history.train_loss[-2]
                            + (
                                self.history.train_loss[-3]
                                - self.history.train_loss[-2]
                            )
                            * self.config.early_stopping_min_delta
                        ):
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1

            if (
                self.config.early_stopping
                and early_stop_counter >= self.config.early_stopping_patience
            ):
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        _ = self.train(False)

        end = time.time()
        self.logger.info(
            f"Training VQ-VAE complete. Time taken: {end - start:.2f} seconds"
        )

        self.logger.info("Training PixelCNN")

        start = time.time()

        for epoch in range(self.config.epochs):
            self.logger.info(f"Training epoch {epoch + 1}/{self.config.epochs}")

        end = time.time()
        self.logger.info(
            f"Training PixelCNN complete. Time taken: {end - start:.2f} seconds"
        )

    @override
    def summary(self, *args, **kwargs):
        pass
