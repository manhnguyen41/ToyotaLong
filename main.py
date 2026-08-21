"""Simple edit-and-run entry point for training and tuning.

Change the constants in the CONFIGURATION section, then run: ``python main.py``.
The src directory is added to sys.path automatically, so an editable package
install is not required (the third-party dependencies are still required).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from service_parts_forecasting.cli.common import run_from_args  # noqa: E402
from service_parts_forecasting.config import load_config  # noqa: E402
from service_parts_forecasting.data.loader import load_actuals  # noqa: E402
from service_parts_forecasting.tuning import run_tuning  # noqa: E402


# ---------------------------------------------------------------------------
# CONFIGURATION: edit these values, then run ``python main.py``.
# ---------------------------------------------------------------------------
MODE = "tune_smoke"  # tune_smoke | tune | validation | test

DATA_PATH = PROJECT_ROOT / "data" / (
    "04_20260724_Historical order data and forecasting results for_8605 service parts.xlsx"
)
TUNING_CONFIG = PROJECT_ROOT / "configs" / "tuning" / "lstm_optuna.yaml"
TRAIN_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "lstm_context12.yaml"

N_TRIALS = 30
OPTUNA_STORAGE: str | None = None
# Resume example:
# OPTUNA_STORAGE = "sqlite:///outputs/tuning/global_lstm_validation.db"


def _check_paths(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required file(s) not found: {missing}")


def _run_tuning(*, smoke_test: bool) -> None:
    _check_paths(DATA_PATH, TUNING_CONFIG)
    config = load_config(TUNING_CONFIG)
    data = load_actuals(DATA_PATH)
    _, result = run_tuning(
        data,
        config,
        smoke_test=smoke_test,
        n_trials_override=2 if smoke_test else N_TRIALS,
        storage_override=OPTUNA_STORAGE,
    )
    print(f"Best config: {result['best_config']}")


def _run_experiment(stage: str) -> None:
    _check_paths(DATA_PATH, TRAIN_CONFIG)
    run_from_args(
        SimpleNamespace(
            config=TRAIN_CONFIG,
            data=DATA_PATH,
            stage=stage,
            smoke_test=False,
        )
    )


def main() -> None:
    if MODE == "tune_smoke":
        _run_tuning(smoke_test=True)
    elif MODE == "tune":
        _run_tuning(smoke_test=False)
    elif MODE == "validation":
        _run_experiment("validation")
    elif MODE == "test":
        _run_experiment("test")
    else:
        raise ValueError(
            f"Unknown MODE={MODE!r}; use tune_smoke, tune, validation, or test"
        )


if __name__ == "__main__":
    main()

