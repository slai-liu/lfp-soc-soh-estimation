"""LSTM used by both terminal-voltage and RFORC-shielded SOC estimators."""
from __future__ import annotations

import torch
from torch import nn


class SocLSTM(nn.Module):
    def __init__(self, input_features: int = 3, hidden_size: int = 30) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return torch.sigmoid(self.output(sequence[:, -1, :]))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
