from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .loader import ActualsData
from .normalization import WindowNormalizer
from .windows import WindowIndex, assert_window_is_causal, sliding_window_indices


def _features(demand: Tensor, dates: pd.DatetimeIndex, calendar_features: bool) -> Tensor:
    channels = [demand]
    if calendar_features:
        month = torch.tensor(dates.month.to_numpy(), dtype=torch.float32)
        angle = 2.0 * torch.pi * month / 12.0
        channels.extend([torch.sin(angle), torch.cos(angle)])
    return torch.stack(channels, dim=-1)


class GlobalWindowDataset(Dataset[dict[str, Any]]):
    """All causal training windows pooled across parts."""

    def __init__(
        self,
        data: ActualsData,
        *,
        history_end: pd.Timestamp,
        context_length: int,
        prediction_length: int,
        normalizer: WindowNormalizer,
        calendar_features: bool = False,
    ) -> None:
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.normalizer = normalizer
        self.calendar_features = calendar_features
        matches = np.flatnonzero(data.dates == pd.Timestamp(history_end))
        if len(matches) != 1:
            raise ValueError(f"History cutoff {history_end:%Y-%m} is not in the data timeline")
        self.history_end = pd.Timestamp(history_end)
        self.indices = sliding_window_indices(
            number_of_parts=len(data.part_ids),
            last_available_index=int(matches[0]),
            context_length=context_length,
            prediction_length=prediction_length,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index: WindowIndex = self.indices[item]
        start = index.origin_index - self.context_length + 1
        target_start = index.origin_index + 1
        target_end = target_start + self.prediction_length
        history_raw = torch.from_numpy(self.data.values[index.part_index, start : index.origin_index + 1].copy())
        target_raw = torch.from_numpy(self.data.values[index.part_index, target_start:target_end].copy())
        history_norm, target_norm, scale = self.normalizer.normalize(history_raw, target_raw)
        assert target_norm is not None
        input_dates = self.data.dates[start : index.origin_index + 1]
        target_dates = self.data.dates[target_start:target_end]
        origin_date = self.data.dates[index.origin_index]
        assert_window_is_causal(input_dates, target_dates, origin_date)
        return {
            "part_id": self.data.part_ids[index.part_index],
            "origin_date": origin_date,
            "input": _features(history_norm, input_dates, self.calendar_features),
            "target_normalized": target_norm,
            "target_raw": target_raw,
            "scale": scale,
            "target_dates": list(target_dates),
        }


class ForecastOriginDataset(Dataset[dict[str, Any]]):
    """One direct forecast window per part at a specified rolling origin."""

    def __init__(
        self,
        data: ActualsData,
        *,
        history_end: pd.Timestamp,
        context_length: int,
        prediction_length: int,
        normalizer: WindowNormalizer,
        calendar_features: bool = False,
    ) -> None:
        self.data = data
        self.history_end = pd.Timestamp(history_end)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.normalizer = normalizer
        self.calendar_features = calendar_features
        matches = np.flatnonzero(data.dates == self.history_end)
        if len(matches) != 1:
            raise ValueError(f"Forecast origin {history_end:%Y-%m} is not in the data timeline")
        self.origin_index = int(matches[0])
        if self.origin_index + prediction_length >= len(data.dates):
            raise ValueError("Forecast targets extend beyond the supplied actual data")
        if self.origin_index - context_length + 1 < 0:
            raise ValueError("Context length exceeds available history")

    def __len__(self) -> int:
        return len(self.data.part_ids)

    def __getitem__(self, part_index: int) -> dict[str, Any]:
        start = self.origin_index - self.context_length + 1
        target_start = self.origin_index + 1
        target_end = target_start + self.prediction_length
        history_raw = torch.from_numpy(self.data.values[part_index, start : self.origin_index + 1].copy())
        target_raw = torch.from_numpy(self.data.values[part_index, target_start:target_end].copy())
        history_norm, target_norm, scale = self.normalizer.normalize(history_raw, target_raw)
        assert target_norm is not None
        input_dates = self.data.dates[start : self.origin_index + 1]
        target_dates = self.data.dates[target_start:target_end]
        assert_window_is_causal(input_dates, target_dates, self.history_end)
        return {
            "part_id": self.data.part_ids[part_index],
            "origin_date": self.history_end,
            "input": _features(history_norm, input_dates, self.calendar_features),
            "target_normalized": target_norm,
            "target_raw": target_raw,
            "scale": scale,
            "target_dates": list(target_dates),
        }


def forecast_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "part_id": [sample["part_id"] for sample in samples],
        "origin_date": [sample["origin_date"] for sample in samples],
        "input": torch.stack([sample["input"] for sample in samples]),
        "target_normalized": torch.stack([sample["target_normalized"] for sample in samples]),
        "target_raw": torch.stack([sample["target_raw"] for sample in samples]),
        "scale": torch.stack([sample["scale"] for sample in samples]),
        "target_dates": [sample["target_dates"] for sample in samples],
    }

