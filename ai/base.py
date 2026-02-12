from typing import Callable, Union

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam

from ai.dataset import Loader
from ai.lstm_model import Model as LSTMModel
from ai.gru_model import Model as GRUModel
from ai.timexer_model import TimeXer
from ai.rnn_model import Model as RNNModel

import matplotlib.pyplot as plt

type Model = Union[RNNModel, LSTMModel, GRUModel, TimeXer]


class Base:

    def __init__(
        self, dataset: Loader, model: Callable[[int], Model], name: str = "BaseModel"
    ):
        self._dataset = dataset
        self._model_callable = model
        self._name = name

    def prepare(self):
        # Encode
        self._dataset.encode()

        # Get the features and targets
        features, _ = self._dataset.get_features_and_targets()

        # Split the dataset into train, validation, and test sets
        self._train_loader, self._validation_loader, self._test_loader = (
            self._dataset.get_loaders(batch_size=32)
        )

        # Initialize the model
        self._model = self._model_callable(features.shape[1])

        # Criterion for regression tasks
        self._criterion = MSELoss()

        # Mean Absolute Error for evaluation
        self._mae_criterion = L1Loss()

        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=0.001)

        return self

    def fit(
        self, epochs: int = 100_000
    ) -> tuple[int, list[float], list[float], list[float], list[float]]:

        # Early stopping parameters
        best_validation_loss = float("inf")
        patience = 30
        current_patience = 0

        # Store the metrics
        avg_train_loss_arr = []
        avg_validation_loss_arr = []
        avg_validation_mae_arr = []
        validation_mape_arr = []

        # Training loop
        for epoch in range(epochs):

            self._model.train()

            # Training loss
            t_loss = 0

            # Iterate over the training data
            for features, target in self._train_loader:

                # Convert to tensors
                if isinstance(features, torch.Tensor):
                    features = features.clone().detach().float()

                else:
                    features = torch.as_tensor(features, dtype=torch.float32)

                if isinstance(target, torch.Tensor):
                    target = target.clone().detach().float()

                else:
                    target = torch.as_tensor(target, dtype=torch.float32)

                # Zero the gradients
                self._optimizer.zero_grad()

                # Forward pass
                output = self._model(features)

                # Compute the loss
                loss = self._criterion(output, target)

                # Backward pass and optimization
                loss.backward()
                self._optimizer.step()

                t_loss += loss.item()

            # Evaluate on the training set
            avg_train_loss = t_loss / len(self._train_loader)

            # Evaluate on the validation set
            avg_validation_loss, avg_validation_mae, validation_mape, _, _ = (
                self.evaluate(model=self._model, loader=self._validation_loader)
            )

            # Store the metrics
            avg_train_loss_arr.append(avg_train_loss)
            avg_validation_loss_arr.append(avg_validation_loss)
            avg_validation_mae_arr.append(avg_validation_mae)
            validation_mape_arr.append(validation_mape)

            # Save the best model based on validation loss
            if avg_validation_loss < best_validation_loss:

                # Update the best validation loss
                best_validation_loss = avg_validation_loss

                # Reset the patience
                current_patience = 0

                # Save the model state, optimizer state, and metrics
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self._model.state_dict(),
                        "optimizer_state_dict": self._optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "validation_loss": avg_validation_loss,
                        "validation_mae": avg_validation_mae,
                        "validation_mape": validation_mape_arr[-1],
                    },
                    f"{self._name}_best_model.pth",
                )

            # Increment patience if validation loss did not improve
            else:
                current_patience += 1

            # Print metrics
            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Validation Loss: {avg_validation_loss:.4f} | "
                f"Validation MAE: {avg_validation_mae:.4f} | "
                f"Validation MAPE: {validation_mape:.2f}% | "
                f"Best Validation Loss: {best_validation_loss:.4f} | "
            )

            # Check for early stopping
            if current_patience >= patience:

                # Update the model with the best model state
                checkpoint = torch.load(f"{self._name}_best_model.pth")
                self._model.load_state_dict(checkpoint["model_state_dict"])
                self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Return the metrics
                return (
                    epoch + 1,
                    avg_train_loss_arr,
                    avg_validation_loss_arr,
                    avg_validation_mae_arr,
                    validation_mape_arr,
                )

        # Update the model with the best model state after training completes
        checkpoint = torch.load(f"{self._name}_best_model.pth")
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Return the metrics
        return (
            epochs + 1,
            avg_train_loss_arr,
            avg_validation_loss_arr,
            avg_validation_mae_arr,
            validation_mape_arr,
        )

    def evaluate(self, model: Model | None = None, loader: DataLoader | None = None):

        # Use the provided model and loader if given, otherwise use the default ones
        model = model if model is not None else self._model
        loader = loader if loader is not None else self._validation_loader

        # Evaluate the model
        model.eval()

        # Metrics
        t_loss = 0
        t_mae = 0

        # Store predictions
        predictions = []
        targets = []

        # No gradients
        with torch.no_grad():

            # Iterate over the validation data
            for features, target in loader:

                # Convert to tensors
                if isinstance(features, torch.Tensor):
                    features = features.clone().detach().float()

                else:
                    features = torch.as_tensor(features, dtype=torch.float32)

                if isinstance(target, torch.Tensor):
                    target = target.clone().detach().float()

                else:
                    target = torch.as_tensor(target, dtype=torch.float32)

                # Forward pass
                output = model(features)

                # Compute the loss and MAE
                t_loss += self._criterion(output, target).item()
                t_mae += self._mae_criterion(output, target).item()

                # Store predictions and targets for MAPE calculation
                predictions.append(output)
                targets.append(target)

        # Concatenate all predictions and targets
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        # Compute the MAPE
        mape = torch.mean(torch.abs((targets - predictions) / targets)) * 100

        # Average the loss and MAE over the dataset
        avg_loss = t_loss / len(loader)
        avg_mae = t_mae / len(loader)

        return avg_loss, avg_mae, mape, predictions, targets

    def test(self):
        # Evaluate the model on the test set
        return self.evaluate(loader=self._test_loader)

    def load(self, path: str):
        # Load the model state, optimizer state, and metrics
        checkpoint = torch.load(path)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    def plot_predictions(
        self, predictions, targets, title: str = "Predictions vs Actual Values"
    ):
        plt.figure(figsize=(14, 7))

        # Line plot instead of scatter for better readability
        plt.plot(
            range(len(targets)),
            targets,
            alpha=0.8,
            label="Actual",
            color="#2E86AB",
            linewidth=2,
            marker="o",
            markersize=4,
            markevery=max(1, len(targets) // 50),
        )
        plt.plot(
            range(len(predictions)),
            predictions,
            alpha=0.8,
            label="Predicted",
            color="#E63946",
            linewidth=2,
            marker="s",
            markersize=4,
            markevery=max(1, len(predictions) // 50),
        )

        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_loss_curve(
        self,
        avg_train_loss_arr,
        avg_validation_loss_arr,
        title: str = "Training and Validation Loss Curves",
    ):
        # Plot training and validation loss curves
        plt.figure(figsize=(14, 7))

        plt.plot(
            avg_train_loss_arr,
            label="Train Loss",
            color="#2E86AB",
            linewidth=2,
            marker="o",
            markersize=4,
            markevery=max(1, len(avg_train_loss_arr) // 50),
        )

        plt.plot(
            avg_validation_loss_arr,
            label="Validation Loss",
            color="#E63946",
            linewidth=2,
            marker="s",
            markersize=4,
            markevery=max(1, len(avg_validation_loss_arr) // 50),
        )

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_mean_absolute_error_curve(
        self,
        avg_validation_mae_arr,
        title: str = "Validation MAE Curve",
    ):
        # Plot validation MAE curve
        plt.figure(figsize=(14, 7))

        plt.plot(
            avg_validation_mae_arr,
            label="Validation MAE",
            color="#E63946",
            linewidth=2,
            marker="s",
            markersize=4,
            markevery=max(1, len(avg_validation_mae_arr) // 50),
        )

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_mean_absolute_percentage_error_curve(
        self,
        validation_mape_arr,
        title: str = "Validation MAPE Curve",
    ):
        # Plot validation MAPE curve
        plt.figure(figsize=(14, 7))

        plt.plot(
            validation_mape_arr,
            label="Validation MAPE",
            color="#E63946",
            linewidth=2,
            marker="s",
            markersize=4,
            markevery=max(1, len(validation_mape_arr) // 50),
        )

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("MAPE (%)")
        plt.legend()
        plt.grid()
        plt.show()
