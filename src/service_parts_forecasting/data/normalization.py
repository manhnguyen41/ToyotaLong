from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class WindowNormalizer(Protocol):
    def normalize(self, history: Tensor, target: Tensor | None = None) -> tuple[Tensor, Tensor | None, Tensor]: ...

    def inverse_target(self, prediction: Tensor, scale: Tensor) -> Tensor: ...


class MeanScaleNormalizer:
    def __init__(self, lookback: int = 12, minimum_scale: float = 1.0) -> None:
        if lookback <= 0 or minimum_scale <= 0:
            raise ValueError("lookback and minimum_scale must be positive")
        self.lookback = lookback
        self.minimum_scale = minimum_scale

    def normalize(
        self, history: Tensor, target: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        recent = history[..., -self.lookback :]
        scale = recent.mean(dim=-1, keepdim=True).clamp_min(self.minimum_scale)
        normalized_target = None if target is None else target / scale
        return history / scale, normalized_target, scale

    def inverse_target(self, prediction: Tensor, scale: Tensor) -> Tensor:
        return prediction * scale


class IdentityNormalizer:
    def normalize(
        self, history: Tensor, target: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        scale = torch.ones((*history.shape[:-1], 1), dtype=history.dtype, device=history.device)
        return history, target, scale

    def inverse_target(self, prediction: Tensor, scale: Tensor) -> Tensor:
        return prediction


def make_normalizer(name: str, lookback: int = 12) -> WindowNormalizer:
    if name == "mean_scale":
        return MeanScaleNormalizer(lookback=lookback)
    if name == "identity":
        return IdentityNormalizer()
    raise ValueError(f"Unknown normalizer {name!r}")

