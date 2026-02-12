from typing import Callable, Union

from ai.base import Base
from ai.dataset import Loader

from ai.lstm_model import Model as LSTMModel
from ai.gru_model import Model as GRUModel
from ai.timexer_model import TimeXer
from ai.rnn_model import Model as RNNModel


class Runner:
    def __init__(
        self,
        path: str,
        model: Callable[[int], Union[RNNModel, LSTMModel, GRUModel, TimeXer]],
        name="Model",
    ):
        self._model = model
        self._path = path
        self._name = name

    def run(self, plot: bool = True) -> tuple:
        # Load the dataset
        dataset = Loader("df_venues_processed.csv")

        # Wrapper
        base = Base(dataset, model=self._model, name=self._name)

        # Prepare the data and model
        base.prepare()

        # Train the model
        (
            epochs,
            avg_train_loss_arr,
            avg_validation_loss_arr,
            avg_validation_mae_arr,
            validation_mape_arr,
        ) = base.fit(epochs=250)
        print(f"Training completed for {self._name} in {epochs} epochs.")

        # Evaluate on the validation set
        evaluation_loss, evaluation_mae, evaluation_mape, _, _ = base.evaluate()

        print(f"\nFinal Evaluation for {self._name} on Validation Set:")
        print(f"  Loss (MSE): {evaluation_loss:.4f}")
        print(f"  MAE: {evaluation_mae:.4f}")
        print(f"  MAPE: {evaluation_mape:.2f}%")

        # Evaluate on the test set
        test_loss, test_mae, test_mape, test_predictions, test_targets = base.test()

        print(f"\nFinal Evaluation for {self._name} on Test Set:")
        print(f"  Loss (MSE): {test_loss:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")

        if not plot:
            return (
                (evaluation_loss, evaluation_mae, evaluation_mape),
                (
                    epochs,
                    avg_train_loss_arr,
                    avg_validation_loss_arr,
                    avg_validation_mae_arr,
                    validation_mape_arr,
                ),
                (test_predictions, test_targets),
            )

        # Plot the training and validation loss curves
        base.plot_loss_curve(
            avg_train_loss_arr,
            avg_validation_loss_arr,
            title=f"{self._name} Training Curves",
        )

        # Plot the validation MAE and MAPE curves
        base.plot_mean_absolute_error_curve(
            avg_validation_mae_arr,
            title=f"{self._name} Validation MAE Curve",
        )

        base.plot_mean_absolute_percentage_error_curve(
            validation_mape_arr,
            title=f"{self._name} Validation MAPE Curve",
        )

        # Plot the predictions vs actual values for the test set
        base.plot_predictions(
            test_predictions,
            test_targets,
            title=f"{self._name} Predictions vs Actual Values on Test Set",
        )

        return (
            (evaluation_loss, evaluation_mae, evaluation_mape),
            (test_loss, test_mae, test_mape),
            (test_predictions, test_targets),
        )
