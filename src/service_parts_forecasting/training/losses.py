from __future__ import annotations

import torch
from torch import Tensor, nn


class NormalizedMAE(nn.Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.abs(prediction - target))


def make_loss(name: str) -> nn.Module:
    if name == "normalized_mae":
        return NormalizedMAE()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss {name!r}")

