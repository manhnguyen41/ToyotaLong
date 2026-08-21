from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from service_parts_forecasting.data.loader import ActualsData


def add_paper_errors(
    predictions: pd.DataFrame, actuals: ActualsData, *, epsilon: float
) -> pd.DataFrame:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    required = {"part_id", "target_date", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    result = predictions.copy()
    result["target_date"] = pd.to_datetime(result["target_date"])
    part_index = {part_id: index for index, part_id in enumerate(actuals.part_ids)}
    date_index = {date: index for index, date in enumerate(actuals.dates)}
    denominators: list[float] = []
    for row in result.itertuples(index=False):
        if row.part_id not in part_index or row.target_date not in date_index:
            raise ValueError(f"Prediction key is absent from actuals: {row.part_id}, {row.target_date}")
        column = date_index[row.target_date]
        if column < 12:
            raise ValueError(f"Paper score requires 12 prior months for {row.target_date:%Y-%m}")
        preceding = actuals.values[part_index[row.part_id], column - 12 : column]
        denominators.append(max(float(np.mean(preceding)), epsilon))
    result["paper_denominator"] = denominators
    result["absolute_error"] = np.abs(result["y_pred"] - result["y_true"])
    result["squared_error"] = np.square(result["y_pred"] - result["y_true"])
    result["paper_error"] = result["absolute_error"] / result["paper_denominator"]
    return result


def _metric_row(frame: pd.DataFrame) -> dict[str, float]:
    actual_sum = float(frame["y_true"].abs().sum())
    return {
        "paper_score": float(frame.groupby("part_id")["paper_error"].mean().mean()),
        "mae": float(frame["absolute_error"].mean()),
        "rmse": float(np.sqrt(frame["squared_error"].mean())),
        "wape": float(frame["absolute_error"].sum() / actual_sum) if actual_sum > 0 else float("nan"),
    }


def compute_metrics(
    predictions: pd.DataFrame, actuals: ActualsData, *, epsilon: float = 1e-8
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scored = add_paper_errors(predictions, actuals, epsilon=epsilon)
    if "seed" not in scored:
        scored["seed"] = 0
    summaries: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    part_frames: list[pd.DataFrame] = []
    for seed, seed_frame in scored.groupby("seed", sort=True):
        metrics = _metric_row(seed_frame)
        per_part = (
            seed_frame.groupby("part_id", sort=False)
            .agg(
                paper_score=("paper_error", "mean"),
                mae=("absolute_error", "mean"),
                rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
                observations=("paper_error", "size"),
            )
            .reset_index()
        )
        per_part.insert(0, "seed", int(seed))
        per_part["score_group"] = np.select(
            [per_part["paper_score"] < 0.3, per_part["paper_score"] <= 0.7],
            ["<0.3", "0.3-0.7"],
            default=">0.7",
        )
        part_frames.append(per_part)
        counts = per_part["score_group"].value_counts()
        summaries.append(
            {
                "seed": int(seed),
                **metrics,
                "parts_score_lt_0_3": int(counts.get("<0.3", 0)),
                "parts_score_0_3_to_0_7": int(counts.get("0.3-0.7", 0)),
                "parts_score_gt_0_7": int(counts.get(">0.7", 0)),
                "parts": int(per_part.shape[0]),
                "predictions": int(seed_frame.shape[0]),
            }
        )
        for horizon, horizon_frame in seed_frame.groupby("horizon", sort=True):
            horizon_rows.append({"seed": int(seed), "horizon": int(horizon), **_metric_row(horizon_frame)})

    summary_frame = pd.DataFrame(summaries)
    aggregate: dict[str, Any] = {
        "epsilon": epsilon,
        "number_of_seeds": int(len(summary_frame)),
        "per_seed": summaries,
    }
    for metric in ("paper_score", "mae", "rmse", "wape"):
        aggregate[f"{metric}_mean"] = float(summary_frame[metric].mean())
        aggregate[f"{metric}_std"] = float(summary_frame[metric].std(ddof=0))
    aggregate["parts_score_lt_0_3_mean"] = float(summary_frame["parts_score_lt_0_3"].mean())
    aggregate["parts_score_0_3_to_0_7_mean"] = float(summary_frame["parts_score_0_3_to_0_7"].mean())
    aggregate["parts_score_gt_0_7_mean"] = float(summary_frame["parts_score_gt_0_7"].mean())
    return aggregate, pd.DataFrame(horizon_rows), pd.concat(part_frames, ignore_index=True)
