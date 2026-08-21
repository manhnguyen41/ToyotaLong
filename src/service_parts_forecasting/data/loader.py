from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

import numpy as np
import pandas as pd

ACTUAL_SHEET = "Actual_Data_8065service-parts"
EXPECTED_PARTS = 8_605
EXPECTED_MONTHS = 62
EXPECTED_START = pd.Timestamp("2017-08-01")
EXPECTED_END = pd.Timestamp("2022-09-01")


@dataclass(frozen=True)
class ActualsData:
    """Canonical actual-demand data and efficient wide representation."""

    long: pd.DataFrame
    part_ids: tuple[str, ...]
    dates: pd.DatetimeIndex
    values: np.ndarray

    def subset_parts(self, count: int | None) -> "ActualsData":
        if count is None or count >= len(self.part_ids):
            return self
        if count <= 0:
            raise ValueError("max_parts must be positive")
        ids = self.part_ids[:count]
        mask = self.long["part_id"].isin(ids)
        return ActualsData(
            long=self.long.loc[mask].reset_index(drop=True),
            part_ids=ids,
            dates=self.dates,
            values=self.values[:count].copy(),
        )


def _parse_month_column(column: Hashable) -> pd.Timestamp | None:
    if isinstance(column, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(column)
    elif hasattr(column, "year") and hasattr(column, "month"):
        timestamp = pd.Timestamp(column)
    else:
        try:
            timestamp = pd.to_datetime(str(column), errors="raise")
        except (ValueError, TypeError):
            return None
    if pd.isna(timestamp):
        return None
    timestamp = pd.Timestamp(timestamp).normalize()
    if timestamp.day != 1:
        raise ValueError(f"Month column {column!r} is not the first day of a month")
    return timestamp


def load_actuals(
    workbook: str | Path,
    *,
    sheet_name: str = ACTUAL_SHEET,
    expected_parts: int | None = EXPECTED_PARTS,
    expected_months: int | None = EXPECTED_MONTHS,
    expected_start: pd.Timestamp | None = EXPECTED_START,
    expected_end: pd.Timestamp | None = EXPECTED_END,
) -> ActualsData:
    """Load and rigorously validate actual demand from the source workbook."""
    path = Path(workbook)
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")
    try:
        frame = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError as exc:
        raise ValueError(f"Workbook does not contain required sheet {sheet_name!r}") from exc
    if "id" not in frame.columns:
        raise ValueError("Actual-data sheet must contain an 'id' column")

    date_pairs = [(column, _parse_month_column(column)) for column in frame.columns]
    date_pairs = [(column, date) for column, date in date_pairs if date is not None]
    if not date_pairs:
        raise ValueError("No monthly demand columns were found")
    parsed_dates = [date for _, date in date_pairs]
    duplicated_dates = pd.DatetimeIndex(parsed_dates)[pd.DatetimeIndex(parsed_dates).duplicated()]
    if len(duplicated_dates):
        raise ValueError(f"Duplicate month columns: {duplicated_dates.strftime('%Y-%m').tolist()}")
    date_pairs.sort(key=lambda pair: pair[1])
    source_columns = [column for column, _ in date_pairs]
    dates = pd.DatetimeIndex([date for _, date in date_pairs], name="date")

    if frame["id"].isna().any() or (frame["id"].astype(str).str.strip() == "").any():
        raise ValueError("Part IDs must not be missing or blank")
    part_ids = frame["id"].astype(str)
    duplicates = part_ids[part_ids.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate part IDs found: {duplicates[:5]}")

    raw_demand = frame.loc[:, source_columns]
    numeric = raw_demand.apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric.isna() & ~raw_demand.isna()
    if invalid_numeric.any().any():
        row, col = np.argwhere(invalid_numeric.to_numpy())[0]
        raise ValueError(
            f"Non-numeric demand for part {part_ids.iloc[row]!r}, month {dates[col]:%Y-%m}"
        )
    if numeric.isna().any().any():
        row, col = np.argwhere(numeric.isna().to_numpy())[0]
        raise ValueError(f"Missing demand for part {part_ids.iloc[row]!r}, month {dates[col]:%Y-%m}")
    values = numeric.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Demand contains non-finite values")
    if (values < 0).any():
        row, col = np.argwhere(values < 0)[0]
        raise ValueError(f"Negative demand for part {part_ids.iloc[row]!r}, month {dates[col]:%Y-%m}")

    if expected_parts is not None and len(part_ids) != expected_parts:
        raise ValueError(f"Expected {expected_parts} parts, found {len(part_ids)}")
    if expected_months is not None and len(dates) != expected_months:
        raise ValueError(f"Expected {expected_months} months, found {len(dates)}")
    if expected_start is not None and dates.min() != pd.Timestamp(expected_start):
        raise ValueError(f"Expected first month {expected_start:%Y-%m}, found {dates.min():%Y-%m}")
    if expected_end is not None and dates.max() != pd.Timestamp(expected_end):
        raise ValueError(f"Expected last month {expected_end:%Y-%m}, found {dates.max():%Y-%m}")
    expected_timeline = pd.date_range(dates.min(), dates.max(), freq="MS")
    if not dates.equals(expected_timeline):
        missing = expected_timeline.difference(dates).strftime("%Y-%m").tolist()
        raise ValueError(f"Monthly timeline is not contiguous; missing months: {missing}")

    long = pd.DataFrame(
        {
            "part_id": np.repeat(part_ids.to_numpy(), len(dates)),
            "date": np.tile(dates.to_numpy(), len(part_ids)),
            "demand": values.reshape(-1),
        }
    )
    return ActualsData(
        long=long,
        part_ids=tuple(part_ids.tolist()),
        dates=dates,
        values=values,
    )

