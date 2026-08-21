from __future__ import annotations

import argparse
import json
from pathlib import Path

from service_parts_forecasting.baseline import compare_reference_baseline
from service_parts_forecasting.data.loader import load_actuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the workbook EEMD-DMD baseline score")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    args = parser.parse_args()
    actuals = load_actuals(args.data)
    summary, comparison, predictions = compare_reference_baseline(
        args.data, actuals, epsilon=args.epsilon
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "baseline_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        comparison.to_csv(args.output_dir / "baseline_score_comparison.csv", index=False)
        predictions.to_csv(args.output_dir / "baseline_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

