from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RollingOrigin:
    block: int
    history_end: pd.Timestamp
    target_dates: tuple[pd.Timestamp, ...]

    @property
    def forecast_origin(self) -> pd.Timestamp:
        return self.history_end


_SCHEDULES = {
    "validation": (
        ("2020-09-01", "2020-10-01"),
        ("2021-01-01", "2021-02-01"),
        ("2021-05-01", "2021-06-01"),
    ),
    "test": (
        ("2021-09-01", "2021-10-01"),
        ("2022-01-01", "2022-02-01"),
        ("2022-05-01", "2022-06-01"),
    ),
}


def get_rolling_origins(stage: str, dates: pd.DatetimeIndex | None = None) -> list[RollingOrigin]:
    if stage not in _SCHEDULES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(_SCHEDULES)}")
    origins = [
        RollingOrigin(
            block=index,
            history_end=pd.Timestamp(history_end),
            target_dates=tuple(pd.date_range(target_start, periods=4, freq="MS")),
        )
        for index, (history_end, target_start) in enumerate(_SCHEDULES[stage], start=1)
    ]
    if dates is not None:
        available = set(pd.DatetimeIndex(dates))
        required = {origin.history_end for origin in origins}
        required.update(date for origin in origins for date in origin.target_dates)
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"Data does not cover {stage} schedule: {[d.strftime('%Y-%m') for d in missing]}")
    return origins

