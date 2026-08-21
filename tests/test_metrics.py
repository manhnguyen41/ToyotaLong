import numpy as np
import pandas as pd

from service_parts_forecasting.data.loader import ActualsData
from service_parts_forecasting.evaluation.metrics import compute_metrics


def test_metrics_match_hand_calculation() -> None:
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")
    values = np.array([[2.0] * 12 + [4.0]], dtype=np.float32)
    long = pd.DataFrame(
        {"part_id": ["p"] * 13, "date": dates, "demand": values.reshape(-1)}
    )
    actuals = ActualsData(long=long, part_ids=("p",), dates=dates, values=values)
    predictions = pd.DataFrame(
        [
            {
                "seed": 52,
                "part_id": "p",
                "target_date": dates[-1],
                "horizon": 1,
                "y_true": 4.0,
                "y_pred": 2.0,
            }
        ]
    )
    summary, by_horizon, by_part = compute_metrics(predictions, actuals)
    assert summary["paper_score_mean"] == 1.0
    assert summary["mae_mean"] == 2.0
    assert summary["rmse_mean"] == 2.0
    assert summary["wape_mean"] == 0.5
    assert by_horizon.loc[0, "paper_score"] == 1.0
    assert by_part.loc[0, "paper_score"] == 1.0

