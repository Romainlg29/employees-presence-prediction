import seaborn as sns
import matplotlib.pyplot as plt


class ModelComparison:

    def __init__(
        self,
        rnn_evaluation_metrics,
        rnn_timexer_metrics,
        rnn_predictions_targets,
        lstm_evaluation_metrics,
        lstm_timexer_metrics,
        lstm_predictions_targets,
        gru_evaluation_metrics,
        gru_timexer_metrics,
        gru_predictions_targets,
        timexer_evaluation_metrics,
        timexer_timexer_metrics,
        timexer_predictions_targets,
    ):

        # Store the evaluation metrics, predictions, and data for each model
        self.rnn_metrics = rnn_evaluation_metrics
        self.rnn_predictions = rnn_predictions_targets
        self.rnn_data = rnn_timexer_metrics

        self.lstm_metrics = lstm_evaluation_metrics
        self.lstm_predictions = lstm_predictions_targets
        self.lstm_data = lstm_timexer_metrics

        self.gru_metrics = gru_evaluation_metrics
        self.gru_predictions = gru_predictions_targets
        self.gru_data = gru_timexer_metrics

        self.timexer_metrics = timexer_evaluation_metrics
        self.timexer_predictions = timexer_predictions_targets
        self.timexer_data = timexer_timexer_metrics

    def _compare_loss_curves(self):
        """Compare training and validation loss curves for all models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        models = [
            ("RNN", self.rnn_data, "#F4A261"),
            ("LSTM", self.lstm_data, "#2E86AB"),
            ("GRU", self.gru_data, "#E63946"),
            ("TimeXer", self.timexer_data, "#06A77D"),
        ]

        # Training Loss
        for name, data, color in models:
            train_loss = data[1]  # avg_train_loss_arr
            ax1.plot(
                train_loss,
                label=name,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
                markevery=max(1, len(train_loss) // 20),
            )

        ax1.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Validation Loss
        for name, data, color in models:
            val_loss = data[2]  # avg_validation_loss_arr
            ax2.plot(
                val_loss,
                label=name,
                color=color,
                linewidth=2,
                marker="s",
                markersize=3,
                markevery=max(1, len(val_loss) // 20),
            )

        ax2.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (MSE)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _compare_mae_curves(self):
        """Compare validation MAE curves for all models"""
        plt.figure(figsize=(14, 7))

        models = [
            ("RNN", self.rnn_data[3], "#F4A261"),
            ("LSTM", self.lstm_data[3], "#2E86AB"),
            ("GRU", self.gru_data[3], "#E63946"),
            ("TimeXer", self.timexer_data[3], "#06A77D"),
        ]

        for name, mae_arr, color in models:
            plt.plot(
                mae_arr,
                label=name,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
                markevery=max(1, len(mae_arr) // 20),
            )

        plt.title("Validation MAE Comparison", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _compare_mape_curves(self):
        """Compare validation MAPE curves for all models"""
        plt.figure(figsize=(14, 7))

        models = [
            ("RNN", self.rnn_data[4], "#F4A261"),
            ("LSTM", self.lstm_data[4], "#2E86AB"),
            ("GRU", self.gru_data[4], "#E63946"),
            ("TimeXer", self.timexer_data[4], "#06A77D"),
        ]

        for name, mape_arr, color in models:
            plt.plot(
                mape_arr,
                label=name,
                color=color,
                linewidth=2,
                marker="s",
                markersize=3,
                markevery=max(1, len(mape_arr) // 20),
            )

        plt.title("Validation MAPE Comparison", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("MAPE (%)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _compare_rmse_curves(self):
        """Compare validation RMSE curves for all models"""
        plt.figure(figsize=(14, 7))

        models = [
            ("RNN", [x**0.5 for x in self.rnn_data[2]], "#F4A261"),
            ("LSTM", [x**0.5 for x in self.lstm_data[2]], "#2E86AB"),
            ("GRU", [x**0.5 for x in self.gru_data[2]], "#E63946"),
            ("TimeXer", [x**0.5 for x in self.timexer_data[2]], "#06A77D"),
        ]

        for name, rmse_arr, color in models:
            plt.plot(
                rmse_arr,
                label=name,
                color=color,
                linewidth=2,
                marker="^",
                markersize=3,
                markevery=max(1, len(rmse_arr) // 20),
            )

        plt.title("Validation RMSE Comparison", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _compare_predictions(self, sample_range=None):
        """Compare predictions vs actual values for all models"""
        plt.figure(figsize=(16, 8))

        # Use first model's targets as reference
        targets = self.lstm_predictions[1]

        if sample_range:
            start, end = sample_range
            targets = targets[start:end]
            x_range = range(start, end)
        else:
            x_range = range(len(targets))

        # Plot actual values
        plt.plot(
            x_range,
            targets,
            label="Actual",
            color="black",
            linewidth=2.5,
            alpha=0.7,
            marker="o",
            markersize=4,
            markevery=max(1, len(targets) // 30),
        )

        # Plot predictions for each model
        models = [
            ("RNN", self.rnn_predictions[0], "#F4A261", "o"),
            ("LSTM", self.lstm_predictions[0], "#2E86AB", "v"),
            ("GRU", self.gru_predictions[0], "#E63946", "s"),
            ("TimeXer", self.timexer_predictions[0], "#06A77D", "^"),
        ]

        for name, predictions, color, marker in models:
            if sample_range:
                predictions = predictions[start:end]
            plt.plot(
                x_range,
                predictions,
                label=f"{name} Predicted",
                color=color,
                linewidth=2,
                alpha=0.8,
                marker=marker,
                markersize=3,
                markevery=max(1, len(predictions) // 30),
            )

        plt.title(
            "Predictions Comparison: All Models vs Actual",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _compare_metrics_bar(self):
        """Compare final evaluation metrics across models using bar charts"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        models = ["RNN", "LSTM", "GRU", "TimeXer"]
        colors = ["#F4A261", "#2E86AB", "#E63946", "#06A77D"]

        # RMSE (Loss)
        rmse_values = [
            self.rnn_metrics[0] ** 0.5,
            self.lstm_metrics[0] ** 0.5,
            self.gru_metrics[0] ** 0.5,
            self.timexer_metrics[0] ** 0.5,
        ]
        ax1.bar(models, rmse_values, color=colors, alpha=0.8)
        ax1.set_title("Root Mean Squared Error (RMSE)", fontweight="bold")
        ax1.set_ylabel("RMSE")
        ax1.grid(axis="y", alpha=0.3)

        # MAE
        mae_values = [
            self.rnn_metrics[1],
            self.lstm_metrics[1],
            self.gru_metrics[1],
            self.timexer_metrics[1],
        ]
        ax2.bar(models, mae_values, color=colors, alpha=0.8)
        ax2.set_title("Mean Absolute Error (MAE)", fontweight="bold")
        ax2.set_ylabel("MAE")
        ax2.grid(axis="y", alpha=0.3)

        # MAPE
        mape_values = [
            self.rnn_metrics[2],
            self.lstm_metrics[2],
            self.gru_metrics[2],
            self.timexer_metrics[2],
        ]
        ax3.bar(models, mape_values, color=colors, alpha=0.8)
        ax3.set_title("Mean Absolute Percentage Error (MAPE)", fontweight="bold")
        ax3.set_ylabel("MAPE (%)")
        ax3.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_all_comparisons(self, predictions_sample_range=None):
        """Plot all comparison charts"""
        print("Generating comparison plots...")
        self._compare_loss_curves()
        self._compare_mae_curves()
        self._compare_mape_curves()
        self._compare_rmse_curves()
        self._compare_metrics_bar()
        self._compare_predictions(sample_range=predictions_sample_range)
        print("All comparison plots generated!")
