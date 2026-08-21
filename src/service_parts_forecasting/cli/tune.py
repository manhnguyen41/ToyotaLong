from __future__ import annotations

import argparse
import json
from pathlib import Path

from service_parts_forecasting.config import load_config
from service_parts_forecasting.data.loader import load_actuals
from service_parts_forecasting.tuning import run_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the global LSTM on rolling validation only")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--storage", help="Optional Optuna storage URL for resumable studies")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_actuals(args.data)
    _, result = run_tuning(
        data,
        config,
        smoke_test=args.smoke_test,
        n_trials_override=args.n_trials,
        storage_override=args.storage,
        output_dir_override=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
