from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from service_parts_forecasting.config import load_config, resolve_runtime_config
from service_parts_forecasting.data.loader import load_actuals
from service_parts_forecasting.evaluation.metrics import compute_metrics
from service_parts_forecasting.evaluation.reporting import create_run_directory, write_run_outputs
from service_parts_forecasting.evaluation.rolling import run_rolling_experiment
from service_parts_forecasting.seed import resolve_device


def add_experiment_arguments(parser: argparse.ArgumentParser, *, default_stage: str) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--stage", choices=("validation", "test"), default=default_stage)
    parser.add_argument("--smoke-test", action="store_true")


def run_from_args(args: argparse.Namespace) -> Path:
    config = resolve_runtime_config(load_config(args.config), smoke_test=args.smoke_test)
    if int(config.get("prediction_length", 4)) != 4:
        raise ValueError("Version 1 experiment requires prediction_length=4")
    if int(config.get("model", {}).get("prediction_length", 4)) != int(config["prediction_length"]):
        raise ValueError("model.prediction_length must equal top-level prediction_length")
    device = resolve_device(str(config.get("device", "auto")))
    data = load_actuals(args.data).subset_parts(config.get("max_parts"))
    run_dir = create_run_directory(config, args.stage)
    predictions, histories, logs = run_rolling_experiment(
        data,
        stage=args.stage,
        config=config,
        device=device,
        checkpoint_dir=run_dir / "checkpoints",
    )
    summary, by_horizon, by_part = compute_metrics(
        predictions, data, epsilon=float(config.get("metric_epsilon", 1e-8))
    )
    summary.update(
        {
            "stage": args.stage,
            "parts": len(data.part_ids),
            "months": len(data.dates),
            "smoke_test": bool(args.smoke_test),
        }
    )
    refit_history = histories.loc[histories["phase"] == "refit_full_history"]
    selected_epochs = (
        refit_history.groupby(["seed", "block"], as_index=False)["epoch"]
        .max()
        .rename(columns={"epoch": "selected_epochs"})
    )
    summary["epochs_by_seed_and_block"] = selected_epochs.to_dict(orient="records")
    if args.stage == "validation":
        summary["suggested_final_epochs_median"] = int(
            np.rint(selected_epochs["selected_epochs"].median())
        )
    write_run_outputs(
        run_dir,
        config=config,
        device=device,
        predictions=predictions,
        histories=histories,
        metrics_summary=summary,
        metrics_by_horizon=by_horizon,
        metrics_by_part=by_part,
        log_lines=logs,
    )
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {run_dir.resolve()}")
    return run_dir
