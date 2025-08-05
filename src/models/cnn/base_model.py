import time
from typing import Any, final, override
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_model import BaseModel
from core.config import CNNConfig
from core.loss import get_loss_function
from core.optimizer import get_optimizer
from core.weights import load_model, save_model


@final
class History:
    train_loss: list[float]
    val_loss: list[float]

    def __init__(self):
        self.train_loss = []
        self.val_loss = []


class BaseCNNModel(BaseModel):
    config: CNNConfig
    history: History

    def __init__(self, config: CNNConfig):
        super(BaseCNNModel, self).__init__("CNN")
        self.config = config

        self.history = History()

    @override
    def save(self, name: str | None = None):
        """Save the model."""
        name = name or self.config.name
        save_model(self, name)

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

            optimizer.zero_grad()
            outputs = self(inputs)

            loss = loss_function(outputs, targets)
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

                outputs = self(inputs)
                loss = loss_function(outputs, targets)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

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
        _ = self.to(self.device)

        optimizer = get_optimizer(self.config.optimizer)(
            self.parameters(), **self.config.optimizer_params.to_kwargs()
        )
        loss_function = get_loss_function(self.config.loss_function).to(self.device)

        early_stop_counter = 0

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
                    if not val_loader:
                        self.logger.warning(
                            "Early stopping is enabled but validation data loader is not provided."
                        )
                        continue

                    if len(self.history.val_loss) < 2:
                        continue

                    if (
                        self.history.val_loss[-1]
                        < self.history.val_loss[-2]
                        + self.config.early_stopping_min_delta
                    ):
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                else:
                    if len(self.history.train_loss) < 2:
                        continue

                    if (
                        self.history.train_loss[-1]
                        < self.history.train_loss[-2]
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

    @override
    def predict(
        self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> Any:
        """Evaluate the model on the provided data loader."""
        _ = self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self(inputs)
                predictions = torch.argmax(outputs, dim=1)

                total += targets.size(0)
                correct += (predictions == targets).sum().item()

        accuracy = (correct / total) * 100
        self.logger.info(f"Test Accuracy: {accuracy:.2f}%")

    def process_output(self, output: torch.Tensor) -> torch.Tensor:
        """Process the model output."""
        return output.cpu()

    @override
    def summary(self, input_size: tuple[int, int, int, int]):
        _ = summary(self, input_size=input_size)

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
