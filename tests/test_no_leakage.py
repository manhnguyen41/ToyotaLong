import pandas as pd
import torch

from service_parts_forecasting.data.dataset import ForecastOriginDataset
from service_parts_forecasting.data.normalization import MeanScaleNormalizer


def test_next_block_uses_observed_actuals_not_previous_predictions(toy_actuals) -> None:
    dataset = ForecastOriginDataset(
        toy_actuals,
        history_end=pd.Timestamp("2021-01-01"),
        context_length=12,
        prediction_length=4,
        normalizer=MeanScaleNormalizer(),
    )
    expected_history = torch.from_numpy(toy_actuals.values[0, 30:42].copy())
    sample = dataset[0]
    recovered_history = sample["input"][:, 0] * sample["scale"]
    assert torch.equal(recovered_history, expected_history)
    assert torch.equal(recovered_history[-4:], torch.tensor([39.0, 40.0, 41.0, 42.0]))

