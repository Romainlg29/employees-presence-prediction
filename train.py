from ai.runner import Runner

from ai.gru_model import Model as GRUModel
from ai.lstm_model import Model as LSTMModel

if __name__ == "__main__":

    lstm_runner = Runner(
        path="df_venues_model.py", model=lambda i: LSTMModel(i), name="LSTM"
    )
    lstm_runner.run(plot=True)

    gru_runner = Runner(
        path="df_venues_model.py", model=lambda i: GRUModel(i), name="GRU"
    )
    gru_runner.run(plot=True)
