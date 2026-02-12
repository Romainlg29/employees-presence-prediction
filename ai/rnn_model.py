import torch
from torch.nn import Module, Linear, ReLU, Sequential, Dropout, RNN


class Model(Module):

    def __init__(
        self,
        input_size=10,
        hidden_size=64,
        num_layers=2,
        dropout_rate=0.2,
        fc_hidden_sizes=[32, 16],
    ):
        super(Model, self).__init__()

        self.rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            nonlinearity="tanh",  # Optional: 'tanh' (default) or 'relu'
        )

        # Fully connected layers after RNN
        fc_layers = []
        prev_size = hidden_size

        for fc_size in fc_hidden_sizes:
            fc_layers.append(Linear(prev_size, fc_size))
            fc_layers.append(ReLU())
            fc_layers.append(Dropout(dropout_rate))
            prev_size = fc_size

        # Output layer
        fc_layers.append(Linear(prev_size, 1))

        self.fc = Sequential(*fc_layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        output = self.fc(last_output)

        return output.squeeze(-1)
