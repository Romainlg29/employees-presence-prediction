import torch
from torch.nn import Module, Linear, ReLU, Sequential, Dropout, BatchNorm1d


class Model(Module):

    def __init__(self, input_size=10, hidden_sizes=[64, 32, 16], dropout_rate=0.2):
        super(Model, self).__init__()

        layers = []

        # Input layer
        layers.append(Linear(input_size, hidden_sizes[0]))
        layers.append(BatchNorm1d(hidden_sizes[0]))
        layers.append(ReLU())
        layers.append(Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(ReLU())
            layers.append(Dropout(dropout_rate))

        # Output layer (single value for regression)
        layers.append(Linear(hidden_sizes[-1], 1))

        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)  # Remove last dimension for scalar output
