from __future__ import annotations

import argparse
import json
from pathlib import Path

from service_parts_forecasting.config import load_config
from service_parts_forecasting.tuning import import_completed_trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Import completed CSV trials into Optuna storage")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--storage", required=True)
    args = parser.parse_args()
    result = import_completed_trials(
        args.results,
        load_config(args.config),
        storage=args.storage,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
