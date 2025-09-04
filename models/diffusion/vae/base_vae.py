from abc import ABC
import time
from typing import override

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import DiffusionConfig
from core.loss import get_loss_function
from core.optimizer import get_optimizer
from core.weights import load_model_old, save_model_old
from ..base_model import DiffusionBaseModel


class VAEBaseModel(DiffusionBaseModel, ABC):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, config: DiffusionConfig):
        super(VAEBaseModel, self).__init__(config)

    @override
    def save(self, name: str | None = None):
        """Save the model."""
        name = name or self.config.name
        save_model_old(self.encoder, f"{name}-encoder")
        save_model_old(self.decoder, f"{name}-decoder")

    @override
    def load(self):
        """Load the model and update weights."""
        loaded_encoder = load_model_old(self.encoder, f"{self.config.name}-encoder")
        loaded_decoder = load_model_old(self.decoder, f"{self.config.name}-decoder")
        if loaded_encoder is None or loaded_decoder is None:
            self.logger.info(f"Model {self.config.name} not found.")
            return
        _ = self.encoder.load_state_dict(loaded_encoder.state_dict())
        _ = self.decoder.load_state_dict(loaded_decoder.state_dict())
        self.logger.info(f"Model {self.config.name} loaded successfully.")

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
        self.logger.info(f"Training complete. Time taken: {end - start:.2f} seconds")

    @override
    def plot_history(self, show: bool, save: bool):
        _ = plt.plot(
            range(1, len(self.history.train_loss) + 1),
            self.history.train_loss,
            marker="o",
            label="Train Loss",
        )
        if len(self.history.val_loss) > 0:
            _ = plt.plot(
                range(1, len(self.history.val_loss) + 1),
                self.history.val_loss,
                marker="o",
                label="Validation Loss",
            )
        _ = plt.title("Training Loss")
        _ = plt.xlabel("Epoch")
        _ = plt.ylabel("Loss")
        plt.grid()
        _ = plt.legend()

        if save:
            plt.savefig(f"images/{self.config.name}_training_loss.png")

        if show:
            plt.show()
