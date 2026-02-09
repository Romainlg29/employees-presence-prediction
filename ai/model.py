import torch
from torch.nn import Module, Linear, ReLU, Sequential


class Model(Module):

    def __init__(self):
        super(Model, self).__init__()

        self.model = Sequential(
            # Input layer
            Linear(input_size, hidden_size),
            ReLU(),
            # Output layer
            Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)
