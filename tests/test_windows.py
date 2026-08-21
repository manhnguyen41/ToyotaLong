import pandas as pd

from service_parts_forecasting.data.dataset import GlobalWindowDataset
from service_parts_forecasting.data.normalization import MeanScaleNormalizer


def test_expected_window_count_and_contract(toy_actuals) -> None:
    dataset = GlobalWindowDataset(
        toy_actuals,
        history_end=pd.Timestamp("2020-09-01"),
        context_length=12,
        prediction_length=4,
        normalizer=MeanScaleNormalizer(),
    )
    assert len(dataset) == 2 * 23
    sample = dataset[-1]
    assert sample["input"].shape == (12, 1)
    assert sample["target_normalized"].shape == (4,)
    assert sample["target_raw"].shape == (4,)
    assert sample["scale"].shape == (1,)
    assert len(sample["target_dates"]) == 4


def test_no_training_window_crosses_cutoff(toy_actuals) -> None:
    cutoff = pd.Timestamp("2020-09-01")
    dataset = GlobalWindowDataset(
        toy_actuals,
        history_end=cutoff,
        context_length=18,
        prediction_length=4,
        normalizer=MeanScaleNormalizer(),
    )
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        assert sample["origin_date"] < min(sample["target_dates"])
        assert max(sample["target_dates"]) <= cutoff

