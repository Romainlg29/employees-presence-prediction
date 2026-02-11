import torch
from torch.nn import Module
from torch.nn import MSELoss
from pytorch_forecasting.models.timexer._timexer_v2 import TimeXer as TimeXerModel


class TimeXer(Module):
    def __init__(self, input_size):
        super().__init__()
        self.timexer = TimeXerModel(
            loss=MSELoss(),
            enc_in=input_size,
            metadata={
                "context_length": 2,
                "prediction_length": 1,
                "feature_indices": {
                    "continuous": list(range(input_size)),
                },
            },
            patch_length=1,
        )

    def forward(self, x):
        # x shape: (batch, features)
        # Reshape to (batch, context_length=2, features) by duplicating
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, 2, 1)  # (batch, 2, features)

        # Create the dictionary input TimeXer expects
        input_dict = {
            "history_cont": x,
        }

        # Get predictions
        output = self.timexer(input_dict)

        # Extract the prediction and reshape to match expected output
        if isinstance(output, dict):
            prediction = output.get("prediction", output.get("output", None))
        else:
            prediction = output

        # Ensure output is (batch,) to match target shape
        if prediction.dim() == 3:
            prediction = prediction.squeeze(-1).squeeze(
                -1
            )  # (batch, pred_len, 1) -> (batch,)
        elif prediction.dim() == 2:
            prediction = prediction.squeeze(-1)  # (batch, 1) -> (batch,)

        return prediction
