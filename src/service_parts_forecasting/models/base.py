from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseForecastModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Return normalized direct forecasts with shape [batch, horizon]."""

