import time
from typing import Any, final, override
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.classification.base_model import History

from ..base_model import BaseModel
from core.config import GANConfig
from core.loss import get_loss_function
from core.optimizer import get_optimizer
from core.weights import load_model, save_model


@final
class GANHistory:
    train_g_loss: list[float]
    train_d_loss: list[float]
    val_g_loss: list[float]
    val_d_loss: list[float]

    def __init__(self):
        self.train_g_loss = []
        self.train_d_loss = []
        self.val_g_loss = []
        self.val_d_loss = []


class GANBaseModel(BaseModel):
    config: GANConfig
    history: GANHistory
    pretrain_history: History
    generator: nn.Module
    discriminator: nn.Module

    def __init__(self, config: GANConfig):
        super(GANBaseModel, self).__init__("GAN")
        self.config = config
        self.history = GANHistory()
        self.pretrain_history = History()

    # @abstractmethod
    # def train_epoch(
    #     self,
    #     train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    #     g_optimizer: optim.Optimizer,
    #     d_optimizer: optim.Optimizer,
    #     g_loss_function: nn.Module,
    #     d_loss_function: nn.Module,
    # ) -> tuple[float, float]:
    #     """Train the model."""
    #     pass

    # @abstractmethod
    # def validate_epoch(
    #     self,
    #     val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    #     g_loss_function: nn.Module,
    #     d_loss_function: nn.Module,
    # ) -> tuple[float, float]:
    #     """Evaluate the model."""
    #     pass

    @override
    def save(self, name: str | None = None):
        """Save the model."""
        name = name or self.config.name
        save_model(self, name)

    def save_discriminator(self, name: str | None = None):
        name = name or f"{self.config.name}_discriminator"
        save_model(self.discriminator, name)

    def save_generator(self, name: str | None = None):
        name = name or f"{self.config.name}_generator"
        save_model(self.generator, name)

    def save_pretrained(self):
        self.save_generator(f"{self.config.name}_pretrained")

    @override
    def load(self):
        """Load the model and update weights."""
        loaded_model = load_model(self, self.config.name)
        if loaded_model is None:
            self.logger.error(f"Model {self.config.name} not found.")
            return
        _ = self.load_state_dict(loaded_model.state_dict())
        _ = self.to(self.device)
        self.logger.info(f"Model {self.config.name} loaded successfully.")

    def load_discriminator(self, name: str | None = None):
        name = name or f"{self.config.name}_discriminator"
        loaded_model = load_model(self.discriminator, name)
        if loaded_model is None:
            self.logger.error(f"Discriminator {name} not found.")
            return
        _ = self.discriminator.load_state_dict(loaded_model.state_dict())
        _ = self.discriminator.to(self.device)
        self.logger.info(f"Discriminator {name} loaded successfully.")

    def load_generator(self, name: str | None = None):
        name = name or f"{self.config.name}_generator"
        loaded_model = load_model(self.generator, name)
        if loaded_model is None:
            self.logger.error(f"Generator {name} not found.")
            return False
        _ = self.generator.load_state_dict(loaded_model.state_dict())
        _ = self.generator.to(self.device)
        self.logger.info(f"Generator {name} loaded successfully.")
        return True

    def load_pretrained(self):
        """Load the pretrained generator."""
        self.logger.info(f"Loading pretrained generator for {self.config.name}")
        return self.load_generator(f"{self.config.name}_pretrained")

    def pretrain_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        optimizer: optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        _ = self.generator.train()
        _ = self.generator.to(self.device)

        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc="Pretraining"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            z = torch.randn(inputs.size(0), 100, device=self.device)

            optimizer.zero_grad()

            g_z = self.generator(z)

            g_loss = loss_function(g_z, targets)

            g_loss.backward()
            optimizer.step()

            epoch_loss += g_loss.item()

        epoch_loss /= len(train_loader)

        _ = self.generator.train(False)

        return epoch_loss

    def pretrain(self, train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]):
        """Pretrain the generator."""
        self.logger.info(
            f"Pretraining {self.config.model} on {self.config.dataset} dataset"
        )
        start = time.time()

        _ = self.generator.to(self.device)

        optimizer = get_optimizer(self.config.pretrain_optimizer)(
            self.generator.parameters(),
            **self.config.pretrain_optimizer_params.to_kwargs(),
        )
        loss_function = get_loss_function(self.config.pretrain_loss_function).to(
            self.device
        )

        for epoch in range(self.config.pretrain_epochs):
            self.logger.info(
                f"Pretraining epoch {epoch + 1}/{self.config.pretrain_epochs}"
            )

            epoch_loss = self.pretrain_epoch(train_loader, optimizer, loss_function)
            self.pretrain_history.train_loss.append(epoch_loss)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.pretrain_epochs} finished, Loss: {epoch_loss:.4f}"
            )

            if (epoch + 1) % self.config.save_after_n_epoch_period == 0:
                self.logger.info(f"Model saved at epoch {epoch + 1}")

        end = time.time()
        self.logger.info(f"Pretraining complete. Time taken: {end - start:.2f} seconds")

    @override
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        g_loss_function: nn.Module,
        d_loss_function: nn.Module,
    ) -> tuple[float, float]:
        """Train the model."""
        _ = self.generator.train()
        _ = self.discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            z = torch.randn(inputs.size(0), 100, device=self.device)

            #######################
            # Train Discriminator #
            #######################

            d_optimizer.zero_grad()

            d_x = self.discriminator(inputs.reshape(inputs.size(0), -1))
            g_z = self.generator(z)
            d_g_z = self.discriminator(g_z)

            d_loss = d_loss_function(d_x, d_g_z)

            d_loss.backward()
            d_optimizer.step()

            #######################
            #   Train Generator   #
            #######################

            g_optimizer.zero_grad()

            g_z = self.generator(z)
            d_g_z = self.discriminator(g_z)

            g_loss = g_loss_function(
                d_g_z,
                torch.full_like(d_g_z, 1.0, device=self.device),
            )

            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        _ = self.generator.train(False)
        _ = self.discriminator.train(False)

        return epoch_g_loss / len(train_loader), epoch_d_loss / len(train_loader)

    @override
    def validate_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        g_loss_function: nn.Module,
        d_loss_function: nn.Module,
    ) -> tuple[float, float]:
        """validate the model."""
        _ = self.eval()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                inputs, _ = batch
                inputs = inputs.to(self.device)

                z = torch.randn(inputs.size(0), 100, device=self.device)
                g_z = self.generator(z)
                d_g_z = self.discriminator(g_z)
                g_loss = g_loss_function(
                    d_g_z,
                    torch.full_like(d_g_z, self.config.real_label, device=self.device),
                )

                d_x = self.discriminator(inputs.reshape(inputs.size(0), -1))
                d_g_z = self.discriminator(g_z)
                d_loss = d_loss_function(d_x, d_g_z)

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

        return epoch_g_loss / len(val_loader), epoch_d_loss / len(val_loader)

    @override
    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        self.logger.info(
            f"Training {self.config.model} on {self.config.dataset} dataset"
        )
        start = time.time()
        _ = self.generator.to(self.device)
        _ = self.discriminator.to(self.device)

        g_optimizer = get_optimizer(self.config.g_optimizer)(
            self.generator.parameters(),
            **self.config.g_optimizer_params.to_kwargs(),
        )
        d_optimizer = get_optimizer(self.config.d_optimizer)(
            self.discriminator.parameters(),
            **self.config.d_optimizer_params.to_kwargs(),
        )
        g_loss_function = get_loss_function(self.config.g_loss_function).to(self.device)
        d_loss_function = get_loss_function(
            self.config.d_loss_function,
            {
                "fake_label": self.config.fake_label,
                "real_label": self.config.real_label,
            },
        ).to(self.device)

        early_stop_counter = 0
        warning_printed = False

        for epoch in range(self.config.epochs):
            self.logger.info(f"Training epoch {epoch + 1}/{self.config.epochs}")

            g_epoch_loss, d_epoch_loss = self.train_epoch(
                train_loader, g_optimizer, d_optimizer, g_loss_function, d_loss_function
            )

            self.history.train_g_loss.append(g_epoch_loss)
            self.history.train_d_loss.append(d_epoch_loss)

            if val_loader:
                epoch_val_g_loss, epoch_val_d_loss = self.validate_epoch(
                    val_loader, g_loss_function, d_loss_function
                )
                self.history.val_g_loss.append(epoch_val_g_loss)
                self.history.val_d_loss.append(epoch_val_d_loss)
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} finished, Train Loss:\n - G: {g_epoch_loss:.4f}\n - D: {d_epoch_loss:.4f}\nVal Loss:\n - G: {epoch_val_g_loss:.4f}\n - D: {epoch_val_d_loss:.4f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} finished, Train Loss:\n - G: {g_epoch_loss:.4f}\n - D: {d_epoch_loss:.4f}"
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

                    if len(self.history.val_g_loss) < 2:
                        continue

                    if self.config.early_stopping_min_delta_strategy == "fixed":
                        if (
                            self.history.val_g_loss[-1]
                            < self.history.val_g_loss[-2]
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
                            self.history.val_g_loss[-1]
                            < self.history.val_g_loss[-2]
                            + self.history.val_g_loss[-2]
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
                            self.history.val_g_loss[-1]
                            < self.history.val_g_loss[-2]
                            + (
                                self.history.val_g_loss[-3]
                                - self.history.val_g_loss[-2]
                            )
                            * self.config.early_stopping_min_delta
                        ):
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                else:
                    if len(self.history.train_g_loss) < 2:
                        continue

                    if self.config.early_stopping_min_delta_strategy == "fixed":
                        if (
                            self.history.train_g_loss[-1]
                            < self.history.train_g_loss[-2]
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
                            self.history.train_g_loss[-1]
                            < self.history.train_g_loss[-2]
                            + self.history.train_g_loss[-2]
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
                            self.history.train_g_loss[-1]
                            < self.history.train_g_loss[-2]
                            + (
                                self.history.train_g_loss[-3]
                                - self.history.train_g_loss[-2]
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

        end = time.time()
        self.logger.info(f"Training complete. Time taken: {end - start:.2f} seconds")

    @override
    def predict(
        self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Any:
        _ = self.discriminator.eval()
        _ = self.generator.eval()

        with torch.no_grad():
            z = torch.randn(128, 100, device=self.device)
            generated_images = self.generator(z)
            generated_images = generated_images.view(-1, 1, 28, 28)

        images = generated_images.cpu()
        # show images using matplotlib
        grid_size = int(images.size(0) ** 0.5)
        _, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < images.size(0):
                    axes[i, j].imshow(images[idx].permute(1, 2, 0).numpy(), cmap="gray")
                axes[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(f"images/{self.config.name}_generated_images.png")
        plt.show()

    def process_output(self, output: torch.Tensor) -> torch.Tensor:
        """Process the model output."""
        return output.cpu()

    @override
    def summary(
        self,
        discriminator_input_size: tuple[int, int, int, int],
        generator_input_size: tuple[int, int, int, int],
    ):
        _ = summary(self.generator, input_size=generator_input_size)
        _ = summary(self.discriminator, input_size=discriminator_input_size)

    def plot_pretrain_history(self, show: bool, save: bool):
        _ = plt.plot(
            range(1, len(self.pretrain_history.train_loss) + 1),
            self.pretrain_history.train_loss,
            marker="o",
            label="Train Loss",
        )
        _ = plt.title("Pretraining Loss")
        _ = plt.xlabel("Epoch")
        _ = plt.ylabel("Loss")
        _ = plt.grid()
        _ = plt.legend()

        if save:
            plt.savefig(f"images/{self.config.name}_pretrain_loss.png")

        if show:
            plt.show()

    @override
    def plot_history(self, show: bool, save: bool):
        _ = plt.plot(
            range(1, len(self.history.train_g_loss) + 1),
            self.history.train_g_loss,
            marker="o",
            label="Train G Loss",
        )
        _ = plt.plot(
            range(1, len(self.history.train_d_loss) + 1),
            self.history.train_d_loss,
            marker="o",
            label="Train D Loss",
        )
        if len(self.history.val_g_loss) > 0:
            _ = plt.plot(
                range(1, len(self.history.val_g_loss) + 1),
                self.history.val_g_loss,
                marker="o",
                label="Validation G Loss",
            )
            _ = plt.plot(
                range(1, len(self.history.val_d_loss) + 1),
                self.history.val_d_loss,
                marker="o",
                label="Validation D Loss",
            )
        _ = plt.title("Training Loss")
        _ = plt.xlabel("Epoch")
        _ = plt.ylabel("Loss")
        _ = plt.grid()
        _ = plt.legend()

        if save:
            plt.savefig(f"images/{self.config.name}_training_loss.png")

        if show:
            plt.show()
