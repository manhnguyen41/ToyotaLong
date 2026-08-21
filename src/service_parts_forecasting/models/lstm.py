from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import BaseForecastModel
from .registry import register_model


@register_model("lstm")
class GlobalLSTM(BaseForecastModel):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        prediction_length: int = 4,
        nonnegative_output: str = "softplus",
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0 or prediction_length <= 0:
            raise ValueError("Model dimensions must be positive")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size * directions, prediction_length)
        if nonnegative_output == "softplus":
            self.output_activation: nn.Module = nn.Softplus()
        elif nonnegative_output in {"none", "identity", None}:
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown output activation {nonnegative_output!r}")

    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        _, (hidden, _) = self.encoder(x)
        if self.encoder.bidirectional:
            encoded = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            encoded = hidden[-1]
        return self.output_activation(self.projection(self.dropout(encoded)))
