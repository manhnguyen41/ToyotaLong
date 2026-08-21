from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from service_parts_forecasting.config import dump_config


def create_run_directory(config: dict[str, Any], stage: str) -> Path:
    model_name = str(config.get("model", {}).get("name", "model"))
    suffix = "smoke" if config.get("smoke_test") else stage
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{suffix}_{uuid4().hex[:8]}"
    path = Path(config.get("output_dir", "outputs")) / model_name / run_id
    (path / "checkpoints").mkdir(parents=True, exist_ok=False)
    return path


def write_environment(path: Path, device: torch.device) -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "requested_device_resolved_to": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (path / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_outputs(
    run_dir: Path,
    *,
    config: dict[str, Any],
    device: torch.device,
    predictions: pd.DataFrame,
    histories: pd.DataFrame,
    metrics_summary: dict[str, Any],
    metrics_by_horizon: pd.DataFrame,
    metrics_by_part: pd.DataFrame,
    log_lines: list[str],
) -> None:
    dump_config(config, run_dir / "resolved_config.yaml")
    write_environment(run_dir, device)
    histories.to_csv(run_dir / "training_history.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    metrics_by_horizon.to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    metrics_by_part.to_csv(run_dir / "metrics_by_part.csv", index=False)
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

