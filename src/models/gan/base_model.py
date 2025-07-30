import logging
import time
from typing import Any, final
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from torchinfo import summary
from torch.utils.data import DataLoader

from core.config import GANConfig
from core.device import get_device
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


class BaseGANModel(nn.Module, metaclass=ABCMeta):
    logger: logging.Logger
    config: GANConfig
    device: torch.device
    history: GANHistory
    generator: nn.Module
    discriminator: nn.Module

    def __init__(self, config: GANConfig):
        super(BaseGANModel, self).__init__()
        self.logger = logging.getLogger("GAN")
        self.config = config
        self.device = get_device()

        self.history = GANHistory()

    @abstractmethod
    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        g_loss_function: nn.Module,
        d_loss_function: nn.Module,
    ) -> tuple[float, float]:
        """Train the model."""
        pass

    @abstractmethod
    def validate_epoch(
        self,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        g_loss_function: nn.Module,
        d_loss_function: nn.Module,
    ) -> tuple[float, float]:
        """Evaluate the model."""
        pass

    def save(self):
        """Save the model."""
        save_model(self, self.config.name)

    def load(self):
        """Load the model and update weights."""
        loaded_model = load_model(self, self.config.name)
        if loaded_model is None:
            self.logger.error(f"Model {self.config.name} not found.")
            return
        _ = self.load_state_dict(loaded_model.state_dict())
        _ = self.to(self.device)
        self.logger.info(f"Model {self.config.name} loaded successfully.")

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
            **self.config.g_optimizer_params,
        )
        d_optimizer = get_optimizer(self.config.d_optimizer)(
            self.discriminator.parameters(),
            **self.config.d_optimizer_params,
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

                    if (
                        self.history.val_g_loss[-1]
                        < self.history.val_g_loss[-2]
                        + self.config.early_stopping_min_delta
                    ):
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                else:
                    if len(self.history.train_g_loss) < 2:
                        continue

                    if (
                        self.history.train_g_loss[-1]
                        < self.history.train_g_loss[-2]
                        + self.config.early_stopping_min_delta
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

    @abstractmethod
    def predict(
        self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Any:
        """Evaluate the model on the provided data loader."""
        pass
        # self.logger.info("Evaluating model...")
        # self.eval()

        # predictions = []
        # with torch.no_grad():
        #     for inputs, targets in tqdm(data_loader, desc="Evaluating"):
        #         inputs, targets = inputs.to(self.device), targets.to(self.device)

        #         outputs = self(inputs)
        #         predictions.append(self.process_output(outputs))

        # self.logger.info("Evaluation complete.")
        # return torch.cat(predictions, dim=0)

    def process_output(self, output: torch.Tensor) -> torch.Tensor:
        """Process the model output."""
        return output.cpu()

    def summary(self, input_size: tuple[int, int, int, int]):
        _ = summary(self.generator, input_size=input_size)
        _ = summary(self.discriminator, input_size=input_size)

    def plot_history(self, show: bool = True, save: bool = True):
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
