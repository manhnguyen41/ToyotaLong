from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd

from service_parts_forecasting.config import dump_config, resolve_runtime_config
from service_parts_forecasting.data.loader import ActualsData
from service_parts_forecasting.evaluation.metrics import compute_metrics
from service_parts_forecasting.evaluation.reporting import write_run_outputs
from service_parts_forecasting.evaluation.rolling import run_rolling_experiment
from service_parts_forecasting.seed import resolve_device


class TrialLike(Protocol):
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int: ...

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False
    ) -> float: ...


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        existing = target.setdefault(key, {})
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {key!r} is not a mapping")
        target = existing
    target[keys[-1]] = value


def sample_search_space(
    trial: TrialLike, base_config: dict[str, Any], search_space: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Sample Optuna parameters and apply dotted keys to a copied experiment config."""
    config = deepcopy(base_config)
    for name, specification in search_space.items():
        kind = str(specification.get("type", "categorical"))
        if kind == "categorical":
            choices = list(specification.get("choices", []))
            if not choices:
                raise ValueError(f"Categorical parameter {name!r} has no choices")
            value = trial.suggest_categorical(name, choices)
        elif kind == "int":
            value = trial.suggest_int(
                name,
                int(specification["low"]),
                int(specification["high"]),
                step=int(specification.get("step", 1)),
                log=bool(specification.get("log", False)),
            )
        elif kind == "float":
            step = specification.get("step")
            value = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                step=None if step is None else float(step),
                log=bool(specification.get("log", False)),
            )
        else:
            raise ValueError(f"Unsupported search-space type {kind!r} for {name!r}")
        _set_nested(config, name, value)
    return config


def _selected_epochs(histories: pd.DataFrame) -> tuple[list[dict[str, int]], int]:
    refits = histories.loc[histories["phase"] == "refit_full_history"]
    by_block = (
        refits.groupby(["seed", "block"], as_index=False)["epoch"]
        .max()
        .rename(columns={"epoch": "selected_epochs"})
    )
    records = [
        {key: int(value) for key, value in row.items()}
        for row in by_block.to_dict(orient="records")
    ]
    return records, int(np.rint(by_block["selected_epochs"].median()))


def _study_directory(output_dir: str | Path, study_name: str, smoke_test: bool) -> Path:
    suffix = "smoke" if smoke_test else "tuning"
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{suffix}_{uuid4().hex[:8]}"
    path = Path(output_dir) / "tuning" / study_name / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_tuning(
    data: ActualsData,
    config: dict[str, Any],
    *,
    smoke_test: bool = False,
    n_trials_override: int | None = None,
    storage_override: str | None = None,
    output_dir_override: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('Optuna is required; install with pip install -e ".[tuning]"') from exc

    if "tuning" not in config:
        raise ValueError("Config has no 'tuning' section")
    tuning_config = deepcopy(config["tuning"])
    base_config = {key: deepcopy(value) for key, value in config.items() if key != "tuning"}
    search_space = tuning_config.get("search_space", {})
    if not search_space:
        raise ValueError("tuning.search_space must not be empty")
    study_name = str(tuning_config.get("study_name", "global_lstm_validation"))
    output_root = output_dir_override or base_config.get("output_dir", "outputs")
    study_dir = _study_directory(output_root, study_name, smoke_test)
    tuning_seeds = [int(seed) for seed in tuning_config.get("tuning_seeds", [52])]
    n_trials = int(n_trials_override or tuning_config.get("n_trials", 30))
    if smoke_test:
        n_trials = min(n_trials, 2)
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    sampler_name = str(tuning_config.get("sampler", "tpe")).lower()
    sampler_seed = int(tuning_config.get("sampler_seed", 52))
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
    else:
        raise ValueError("tuning.sampler must be 'tpe' or 'random'")
    storage = storage_override if storage_override is not None else tuning_config.get("storage")
    study = optuna.create_study(
        study_name=study_name,
        direction=str(tuning_config.get("direction", "minimize")),
        sampler=sampler,
        storage=storage,
        load_if_exists=bool(storage),
    )
    metric_name = str(tuning_config.get("metric", "paper_score_mean"))

    def objective(trial: Any) -> float:
        trial_config = sample_search_space(trial, base_config, search_space)
        trial_config["seeds"] = tuning_seeds
        trial_config = resolve_runtime_config(trial_config, smoke_test=smoke_test)
        trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
        checkpoint_dir = trial_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        device = resolve_device(str(trial_config.get("device", "auto")))
        trial_data = data.subset_parts(trial_config.get("max_parts"))
        predictions, histories, logs = run_rolling_experiment(
            trial_data,
            stage="validation",
            config=trial_config,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        summary, by_horizon, by_part = compute_metrics(
            predictions,
            trial_data,
            epsilon=float(trial_config.get("metric_epsilon", 1e-8)),
        )
        epochs_by_block, suggested_epochs = _selected_epochs(histories)
        summary.update(
            {
                "stage": "validation",
                "trial": int(trial.number),
                "parts": len(trial_data.part_ids),
                "smoke_test": smoke_test,
                "epochs_by_seed_and_block": epochs_by_block,
                "suggested_final_epochs_median": suggested_epochs,
            }
        )
        write_run_outputs(
            trial_dir,
            config=trial_config,
            device=device,
            predictions=predictions,
            histories=histories,
            metrics_summary=summary,
            metrics_by_horizon=by_horizon,
            metrics_by_part=by_part,
            log_lines=logs,
        )
        if metric_name not in summary or not np.isfinite(summary[metric_name]):
            raise FloatingPointError(f"Trial metric {metric_name!r} is missing or non-finite")
        trial.set_user_attr("suggested_final_epochs", suggested_epochs)
        trial.set_user_attr("trial_directory", str(trial_dir.resolve()))
        trial.report(float(summary[metric_name]), step=0)
        return float(summary[metric_name])

    timeout = tuning_config.get("timeout_seconds")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None if timeout is None else float(timeout),
        n_jobs=int(tuning_config.get("n_jobs", 1)),
        gc_after_trial=True,
    )
    if not study.best_trial:
        raise RuntimeError("Tuning completed without a successful trial")

    best_config = deepcopy(base_config)
    for name, value in study.best_trial.params.items():
        _set_nested(best_config, name, value)
    best_config["final_epochs"] = int(study.best_trial.user_attrs["suggested_final_epochs"])
    best_config["seeds"] = [
        int(seed) for seed in tuning_config.get("final_seeds", [52, 62, 72, 82, 92])
    ]
    best_config_name = "smoke_best_config.yaml" if smoke_test else "best_config.yaml"
    dump_config(best_config, study_dir / best_config_name)

    trial_rows: list[dict[str, Any]] = []
    for trial in study.trials:
        duration = None if trial.duration is None else trial.duration.total_seconds()
        trial_rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "duration_seconds": duration,
                **{f"param_{key}": value for key, value in trial.params.items()},
                **{f"attr_{key}": value for key, value in trial.user_attrs.items()},
            }
        )
    pd.DataFrame(trial_rows).to_csv(study_dir / "tuning_results.csv", index=False)
    result = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "metric": metric_name,
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_config": str((study_dir / best_config_name).resolve()),
        "study_directory": str(study_dir.resolve()),
        "completed_trials": sum(trial.state.name == "COMPLETE" for trial in study.trials),
        "smoke_test": smoke_test,
    }
    (study_dir / "tuning_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return study_dir, result

