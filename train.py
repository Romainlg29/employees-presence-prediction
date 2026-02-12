from ai.runner import Runner

from ai.gru_model import Model as GRUModel
from ai.lstm_model import Model as LSTMModel
from ai.timexer_model import TimeXer
from ai.random_forest import RandomForestModel
from ai.rnn_model import Model as RNNModel

from plots.models_compare import ModelComparison


if __name__ == "__main__":

    random_forest = RandomForestModel()
    rf_metrics, rf_predictions, rf_data = random_forest.run_random_forest()
    # random_forest.plot_rf(rf_metrics, rf_predictions, rf_data)

    rnn_runner = Runner(
        path="df_venues_model.py", model=lambda i: RNNModel(i), name="RNN"
    )
    rnn_evaluation_metrics, rnn_timexer_metrics, rnn_predictions_targets = (
        rnn_runner.run(plot=False)
    )

    lstm_runner = Runner(
        path="df_venues_model.py", model=lambda i: LSTMModel(i), name="LSTM"
    )
    lstm_evaluation_metrics, lstm_timexer_metrics, lstm_predictions_targets = (
        lstm_runner.run(plot=False)
    )

    gru_runner = Runner(
        path="df_venues_model.py", model=lambda i: GRUModel(i), name="GRU"
    )
    gru_evaluation_metrics, gru_timexer_metrics, gru_predictions_targets = (
        gru_runner.run(plot=False)
    )

    timexer_runner = Runner(
        path="df_venues_model.py", model=lambda i: TimeXer(i), name="TimeXer"
    )
    timexer_evaluation_metrics, timexer_timexer_metrics, timexer_predictions_targets = (
        timexer_runner.run(plot=False)
    )

    compare = ModelComparison(
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
    )

    compare.plot_all_comparisons()
