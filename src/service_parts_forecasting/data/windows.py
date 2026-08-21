from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowIndex:
    part_index: int
    origin_index: int


def sliding_window_indices(
    *,
    number_of_parts: int,
    last_available_index: int,
    context_length: int,
    prediction_length: int,
) -> list[WindowIndex]:
    """Return pooled indices whose complete target is available by the cutoff."""
    first_origin = context_length - 1
    last_origin = last_available_index - prediction_length
    if last_origin < first_origin:
        raise ValueError(
            "Not enough history for context_length + prediction_length: "
            f"available={last_available_index + 1}, required={context_length + prediction_length}"
        )
    return [
        WindowIndex(part_index=part, origin_index=origin)
        for part in range(number_of_parts)
        for origin in range(first_origin, last_origin + 1)
    ]


def assert_window_is_causal(
    input_dates: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex,
    forecast_origin: pd.Timestamp,
) -> None:
    if len(input_dates) == 0 or len(target_dates) == 0:
        raise AssertionError("Input and target dates must not be empty")
    if input_dates.max() > forecast_origin:
        raise AssertionError("Input contains a date after the forecast origin")
    if target_dates.min() <= forecast_origin:
        raise AssertionError("Target contains a date on or before the forecast origin")

