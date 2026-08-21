import torch

from service_parts_forecasting.data.normalization import MeanScaleNormalizer


def test_normalization_uses_only_last_twelve_history_values() -> None:
    normalizer = MeanScaleNormalizer(lookback=12)
    history = torch.cat([torch.tensor([10_000.0]), torch.arange(1.0, 13.0)])
    target_a = torch.tensor([2.0, 4.0, 6.0, 8.0])
    target_b = torch.tensor([20_000.0, 40_000.0, 60_000.0, 80_000.0])
    _, _, scale_a = normalizer.normalize(history, target_a)
    _, _, scale_b = normalizer.normalize(history, target_b)
    assert torch.equal(scale_a, scale_b)
    assert torch.allclose(scale_a, torch.tensor([6.5]))


def test_inverse_recovers_target() -> None:
    normalizer = MeanScaleNormalizer()
    history = torch.arange(1.0, 13.0)
    target = torch.tensor([3.0, 5.0, 7.0, 9.0])
    _, normalized, scale = normalizer.normalize(history, target)
    assert normalized is not None
    assert torch.allclose(normalizer.inverse_target(normalized, scale), target)


def test_zero_history_is_finite() -> None:
    normalizer = MeanScaleNormalizer()
    history = torch.zeros(12)
    target = torch.zeros(4)
    normalized_history, normalized_target, scale = normalizer.normalize(history, target)
    assert scale.item() == 1.0
    assert torch.isfinite(normalized_history).all()
    assert normalized_target is not None and torch.isfinite(normalized_target).all()

