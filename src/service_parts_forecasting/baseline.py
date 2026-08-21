from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from service_parts_forecasting.data.loader import ActualsData
from service_parts_forecasting.evaluation.metrics import compute_metrics

BASELINE_SHEET = "Proposed method_ITISE paper"


def load_reference_predictions(
    workbook: str | Path, actuals: ActualsData
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse baseline forecasts without relying on the mojibake Japanese headers."""
    frame = pd.read_excel(workbook, sheet_name=BASELINE_SHEET, engine="openpyxl")
    target_dates = pd.date_range("2021-10-01", "2022-09-01", freq="MS")
    date_column_map: dict[pd.Timestamp, object] = {}
    for column in frame.columns:
        try:
            parsed = pd.Timestamp(column).normalize()
        except (TypeError, ValueError):
            continue
        if parsed in target_dates:
            date_column_map[parsed] = column
    if set(date_column_map) != set(target_dates):
        missing = target_dates.difference(pd.DatetimeIndex(date_column_map)).strftime("%Y-%m").tolist()
        raise ValueError(f"Reference sheet is missing forecast months: {missing}")

    non_date = [column for column in frame.columns if column not in date_column_map.values()]
    expected_ids = set(actuals.part_ids)
    id_candidates: list[tuple[object, int]] = []
    for column in non_date:
        matches = frame[column].astype(str).isin(expected_ids).sum()
        if matches:
            id_candidates.append((column, int(matches)))
    if not id_candidates:
        raise ValueError("Could not identify the part-ID column in the reference sheet")
    id_column = max(id_candidates, key=lambda item: item[1])[0]

    score_candidates: list[object] = []
    for column in non_date:
        if column == id_column or str(column).startswith("Unnamed"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == len(actuals.part_ids):
            score_candidates.append(column)
    if not score_candidates:
        raise ValueError("Could not identify the score column in the reference sheet")
    score_column = score_candidates[0]

    frame = frame.loc[frame[id_column].astype(str).isin(expected_ids)].copy()
    if frame[id_column].duplicated().any() or len(frame) != len(actuals.part_ids):
        raise ValueError("Reference predictions do not contain exactly one row per part")
    actual_lookup = actuals.long.set_index(["part_id", "date"])["demand"]
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False, name=None):
        values = dict(zip(frame.columns, row))
        part_id = str(values[id_column])
        for month_index, target_date in enumerate(target_dates):
            prediction = pd.to_numeric(values[date_column_map[target_date]], errors="coerce")
            if pd.isna(prediction) or not np.isfinite(prediction):
                raise ValueError(f"Invalid reference prediction for {part_id}, {target_date:%Y-%m}")
            block = month_index // 4 + 1
            forecast_origin = target_dates[(block - 1) * 4] - pd.DateOffset(months=1)
            rows.append(
                {
                    "stage": "test_reference",
                    "block": block,
                    "seed": 0,
                    "part_id": part_id,
                    "forecast_origin": forecast_origin,
                    "target_date": target_date,
                    "horizon": month_index % 4 + 1,
                    "y_true": float(actual_lookup.loc[(part_id, target_date)]),
                    "y_pred": float(prediction),
                    "y_pred_normalized": np.nan,
                    "scale": np.nan,
                }
            )
    supplied_scores = frame[[id_column, score_column]].rename(
        columns={id_column: "part_id", score_column: "workbook_score"}
    )
    supplied_scores["part_id"] = supplied_scores["part_id"].astype(str)
    supplied_scores["workbook_score"] = pd.to_numeric(supplied_scores["workbook_score"], errors="raise")
    return pd.DataFrame(rows), supplied_scores


def compare_reference_baseline(
    workbook: str | Path, actuals: ActualsData, *, epsilon: float = 1e-8
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    predictions, supplied = load_reference_predictions(workbook, actuals)
    summary, _, recomputed = compute_metrics(predictions, actuals, epsilon=epsilon)
    comparison = supplied.merge(
        recomputed.loc[:, ["part_id", "paper_score"]].rename(columns={"paper_score": "recomputed_score"}),
        on="part_id",
        validate="one_to_one",
    )
    comparison["absolute_discrepancy"] = (
        comparison["workbook_score"] - comparison["recomputed_score"]
    ).abs()
    summary["workbook_score_mean"] = float(comparison["workbook_score"].mean())
    summary["max_part_score_discrepancy"] = float(comparison["absolute_discrepancy"].max())
    summary["parts_with_discrepancy_gt_1e_6"] = int((comparison["absolute_discrepancy"] > 1e-6).sum())
    return summary, comparison, predictions

