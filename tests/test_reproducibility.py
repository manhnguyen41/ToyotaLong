from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from service_parts_forecasting.data.dataset import GlobalWindowDataset, forecast_collate
from service_parts_forecasting.data.normalization import MeanScaleNormalizer
from service_parts_forecasting.models.lstm import GlobalLSTM
from service_parts_forecasting.seed import seed_everything
from service_parts_forecasting.training.trainer import Trainer


def _one_epoch(toy_actuals, checkpoint: Path) -> tuple[float, dict[str, torch.Tensor]]:
    seed_everything(52)
    dataset = GlobalWindowDataset(
        toy_actuals,
        history_end=pd.Timestamp("2020-09-01"),
        context_length=12,
        prediction_length=4,
        normalizer=MeanScaleNormalizer(),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=forecast_collate)
    model = GlobalLSTM(input_size=1, hidden_size=8, num_layers=1, dropout=0.0)
    config = {
        "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "early_stopping": {"enabled": False},
        "gradient_clip_norm": 1.0,
        "loss": {"name": "normalized_mae"},
    }
    result = Trainer(config, torch.device("cpu")).fit(
        model, loader, epochs=1, checkpoint_path=checkpoint
    )
    return result.best_loss, {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_fixed_seed_reproduces_smoke_training(toy_actuals, tmp_path) -> None:
    loss_a, state_a = _one_epoch(toy_actuals, tmp_path / "a.pt")
    loss_b, state_b = _one_epoch(toy_actuals, tmp_path / "b.pt")
    assert loss_a == loss_b
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)
