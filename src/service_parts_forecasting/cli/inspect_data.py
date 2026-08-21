from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from service_parts_forecasting.data.loader import load_actuals
from service_parts_forecasting.data.splits import get_rolling_origins


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the service-parts workbook")
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    data = load_actuals(args.data)
    payload = {
        "parts": len(data.part_ids),
        "months": len(data.dates),
        "start": data.dates.min().strftime("%Y-%m"),
        "end": data.dates.max().strftime("%Y-%m"),
        "missing_values": int(np.isnan(data.values).sum()),
        "minimum_demand": float(data.values.min()),
        "maximum_demand": float(data.values.max()),
        "validation_blocks": [
            {
                "block": item.block,
                "history_end": item.history_end.strftime("%Y-%m"),
                "targets": [date.strftime("%Y-%m") for date in item.target_dates],
            }
            for item in get_rolling_origins("validation", data.dates)
        ],
        "test_blocks": [
            {
                "block": item.block,
                "history_end": item.history_end.strftime("%Y-%m"),
                "targets": [date.strftime("%Y-%m") for date in item.target_dates],
            }
            for item in get_rolling_origins("test", data.dates)
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

