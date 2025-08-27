from typing import Any, Literal, final, override

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from einops import rearrange
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import DiffusionConfig
from models.diffusion.base_model import DiffusionBaseModel


def Normalize(in_channels: int, groups: int = 32):
    return nn.GroupNorm(groups, in_channels, 1e-6, True)


@final
class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(AttentionBlock, self).__init__()

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.norm(x)

        q: torch.Tensor = self.q(o)
        k: torch.Tensor = self.k(o)
        v: torch.Tensor = self.v(o)

        batch, channel, height, width = x.shape
        hw = height * width

        q = q.reshape(batch, channel, hw)
        q = q.permute(0, 2, 1)  # b, hw, c
        k = k.reshape(batch, channel, hw)  # b, c, hw

        # 1. batched matmul
        w = torch.bmm(q, k)  # b,hw(q),hw(k)    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # 2. scale
        w = w * (channel**-0.5)
        # 3. softmax
        w = F.softmax(w, dim=2)

        v = v.reshape(batch, channel, hw)
        w = w.permute(0, 2, 1)  # b, hw(k), hw(q)
        # 4. batched matmul
        o = torch.bmm(v, w)  # b, c, hw(q)      sum_i v[b,c,i] w_[b,i,j]
        o = o.reshape(batch, channel, width, height)

        o = self.proj_out(o)
        o = o + x

        return o


@final
class LinearAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, heads: int = 4, dim_heads: int = 32):
        super(LinearAttentionBlock, self).__init__()

        hidden_channels = heads * dim_heads

        self.heads = heads

        self.qkv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels * 3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.out = nn.Conv2d(
            in_channels=hidden_channels, out_channels=in_channels, kernel_size=1
        )

    @override
    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        o = torch.einsum("bhde,bhdn->bhen", context, q)
        o = rearrange(
            o, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        o = self.out(o)

        return o


AttentionType = Literal["vanilla", "linear", "none"]


def create_attention_block(in_channels: int, type: AttentionType):
    match type:
        case "none":
            return nn.Identity(in_channels)
        case "vanilla":
            return AttentionBlock(in_channels)
        case "linear":
            return LinearAttentionBlock(in_channels, 1, in_channels)


@final
class EncoderDownSampleBlock(nn.Module):
    def __init__(self):
        super(EncoderDownSampleBlock, self).__init__()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = x

        return x


@final
class EncoderResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, temb_channels: int, dropout: float
    ):
        super(EncoderResidualBlock, self).__init__()

        self.layer = nn.Sequential()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.layer(x)

        o = o + x

        return o


@final
class EncoderDownBlock(nn.Module):
    def __init__(
        self,
        blocks: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float,
        use_attention: bool,
        use_downsample: bool,
        attention_type: AttentionType,
    ):
        super(EncoderDownBlock, self).__init__()

        self.blocks = blocks
        self.use_attention = use_attention
        self.use_downsample = use_downsample

        self.residual_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for _ in range(blocks):
            _ = self.residual_blocks.append(
                EncoderResidualBlock(in_channels, out_channels, temb_channels, dropout)
            )
            in_channels = out_channels

            if use_attention:
                _ = self.attention_blocks.append(
                    create_attention_block(in_channels, attention_type)
                )

        if use_downsample:
            self.downsample = EncoderDownSampleBlock()

    @override
    def forward(self, hs: list[torch.Tensor], temb: Any) -> list[torch.Tensor]:
        for i in range(self.blocks):
            h = self.residual_blocks[i](hs[-1], temb)
            if self.use_attention:
                h = self.attention_blocks[i](h)
            hs.append(h)

        if self.use_downsample:
            h = self.downsample(hs[-1])
            hs.append(h)

        return h


@final
class Encoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int = 3,
        ch: int = 128,
        channel_multiplier: list[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        attention_image_sizes: list[int] = [],
    ):
        super(Encoder, self).__init__()

        self.temb_ch = 0

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        in_channel_multipliers = [1]
        in_channel_multipliers.extend(channel_multiplier)

        current_image_size = image_size

        for i in range(len(channel_multiplier)):
            block_in_channels = ch * in_channel_multipliers[i]
            block_out_channels = ch * in_channel_multipliers[i + 1]

            blocks = nn.ModuleList()
            attentions = nn.ModuleList()

            for _ in range(num_res_blocks):
                _ = blocks.append(
                    EncoderResidualBlock(
                        block_in_channels, block_out_channels, self.temb_ch, dropout
                    )
                )
                block_in_channels = block_out_channels

                # if current_image_size in attention_image_sizes:
                # attentions.append()

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.conv1(x)

        mu = self.fc_mu(o)

        logvar = self.fc_logvar(o)
        sigma = torch.exp(logvar * 0.5)

        return mu, sigma


@final
class Decoder(nn.Module):
    def __init__(
        self, latent_channels: int, hidden_channels: int, out_channels: int
    ) -> None:
        super(Decoder, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels, bias=False),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=False)

    @override
    def forward(self, x: torch.Tensor):
        o = self.fc1(x)
        o = self.fc2(o)

        return o


@final
class VAE(DiffusionBaseModel):
    def __init__(self, config: DiffusionConfig):
        super(VAE, self).__init__(config)

        if self.config.dataset == "mnist":
            self.image_size = 28
        elif self.config.dataset == "cifar10" or self.config.dataset == "cifar100":
            self.image_size = 32
        elif (
            self.config.dataset == "imagenet" or self.config.dataset == "mini_imagenet"
        ):
            self.image_size = 224
        else:
            self.image_size = 32

        self.in_channels = self.image_size**2
        self.hidden_channels = 200
        self.latent_channels = 20

        self.encoder = Encoder(
            self.in_channels, self.hidden_channels, self.latent_channels
        )
        self.decoder = Decoder(
            self.latent_channels, self.hidden_channels, self.in_channels
        )

    @override
    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        x_hat = self.decoder(z)

        return x_hat

    @override
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        """Train the model."""
        _ = self.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            x = inputs.view((-1, self.in_channels))

            optimizer.zero_grad()
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps

            x_hat = self.decoder(z)

            loss = loss_function(x, x_hat, mu, sigma)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        _ = self.train(False)

        return epoch_loss / len(train_loader)

    @override
    def validate_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        loss_function: nn.Module,
    ) -> float:
        """Evaluate the model."""
        _ = self.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                x = inputs.view((-1, self.in_channels))

                mu, sigma = self.encoder(x)
                eps = torch.randn_like(sigma)
                z = mu + sigma * eps

                x_hat = self.decoder(z)
                loss = loss_function(x, x_hat, mu, sigma)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    @override
    def predict(
        self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Any:
        """Evaluate the model on the provided data loader."""
        _ = self.eval()

        batch_size = 64

        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_channels, device=self.device)

            o: torch.Tensor = self.decoder(z)
            o = o.clamp(0, 1)
            o = o.view((batch_size, 1, self.image_size, self.image_size))

        grid_img = torchvision.utils.make_grid(o, nrow=8, padding=8, normalize=True)
        _ = plt.imshow(grid_img.permute(1, 2, 0))
        _ = plt.axis("off")
        _ = plt.show()

    @override
    def summary(self, input_size: tuple[int, int, int, int]):
        self.logger.info("Encoder summary")
        _ = summary(self.encoder, (self.in_channels,))
        self.logger.info("Decoder summary")
        _ = summary(self.decoder, (self.latent_channels,))
