from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from service_parts_forecasting.data.dataset import (
    ForecastOriginDataset,
    GlobalWindowDataset,
    forecast_collate,
)
from service_parts_forecasting.data.loader import ActualsData
from service_parts_forecasting.data.normalization import make_normalizer
from service_parts_forecasting.data.splits import RollingOrigin, get_rolling_origins
from service_parts_forecasting.models import create_model
from service_parts_forecasting.seed import seed_everything
from service_parts_forecasting.training import Trainer


def _loader(dataset: torch.utils.data.Dataset, config: dict[str, Any], *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 256)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=forecast_collate,
    )


def _new_model(config: dict[str, Any]) -> torch.nn.Module:
    input_size = 3 if bool(config.get("calendar_features", False)) else 1
    return create_model(config["model"], input_size=input_size)


def _train_model_at_origin(
    data: ActualsData,
    origin: RollingOrigin,
    *,
    stage: str,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    warm_state: dict[str, torch.Tensor] | None,
) -> tuple[torch.nn.Module, list[dict[str, Any]], int]:
    normalizer = make_normalizer(
        str(config.get("normalizer", "mean_scale")), int(config.get("normalization_lookback", 12))
    )
    context = int(config["context_length"])
    horizon = int(config.get("prediction_length", 4))
    calendar = bool(config.get("calendar_features", False))
    trainer = Trainer(config, device)
    history: list[dict[str, Any]] = []

    selected_epochs = int(config.get("final_epochs", config.get("epochs", 100)))
    if stage == "validation" and bool(config.get("early_stopping", {}).get("enabled", True)):
        selection_origin = origin.history_end - pd.DateOffset(months=horizon)
        selection_train = GlobalWindowDataset(
            data,
            history_end=selection_origin,
            context_length=context,
            prediction_length=horizon,
            normalizer=normalizer,
            calendar_features=calendar,
        )
        selection_validation = ForecastOriginDataset(
            data,
            history_end=selection_origin,
            context_length=context,
            prediction_length=horizon,
            normalizer=normalizer,
            calendar_features=calendar,
        )
        selection_model = _new_model(config)
        selection_result = trainer.fit(
            selection_model,
            _loader(selection_train, config, shuffle=True),
            epochs=int(config.get("epochs", 100)),
            checkpoint_path=checkpoint_dir / f"seed_{seed}_block_{origin.block}_selection.pt",
            validation_loader=_loader(selection_validation, config, shuffle=False),
            phase="epoch_selection",
        )
        selected_epochs = selection_result.best_epoch
        history.extend(selection_result.history)

    seed_everything(seed)
    model = _new_model(config)
    if warm_state is not None:
        model.load_state_dict(warm_state)
    full_train = GlobalWindowDataset(
        data,
        history_end=origin.history_end,
        context_length=context,
        prediction_length=horizon,
        normalizer=normalizer,
        calendar_features=calendar,
    )
    fit_result = trainer.fit(
        model,
        _loader(full_train, config, shuffle=True),
        epochs=selected_epochs,
        checkpoint_path=checkpoint_dir / f"seed_{seed}_block_{origin.block}.pt",
        validation_loader=None,
        phase="refit_full_history",
    )
    history.extend(fit_result.history)
    for row in history:
        row.update({"stage": stage, "seed": seed, "block": origin.block})
    return model, history, selected_epochs


def _predict_origin(
    model: torch.nn.Module,
    data: ActualsData,
    origin: RollingOrigin,
    *,
    stage: str,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    normalizer = make_normalizer(
        str(config.get("normalizer", "mean_scale")), int(config.get("normalization_lookback", 12))
    )
    dataset = ForecastOriginDataset(
        data,
        history_end=origin.history_end,
        context_length=int(config["context_length"]),
        prediction_length=int(config.get("prediction_length", 4)),
        normalizer=normalizer,
        calendar_features=bool(config.get("calendar_features", False)),
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in _loader(dataset, config, shuffle=False):
            normalized = model(batch["input"].to(device)).cpu()
            raw = normalizer.inverse_target(normalized, batch["scale"])
            for sample_index, part_id in enumerate(batch["part_id"]):
                for horizon_index, target_date in enumerate(batch["target_dates"][sample_index]):
                    if target_date <= origin.history_end:
                        raise AssertionError("Forecast target is not after its origin")
                    rows.append(
                        {
                            "stage": stage,
                            "block": origin.block,
                            "seed": seed,
                            "part_id": part_id,
                            "forecast_origin": origin.history_end,
                            "target_date": target_date,
                            "horizon": horizon_index + 1,
                            "y_true": float(batch["target_raw"][sample_index, horizon_index]),
                            "y_pred": float(raw[sample_index, horizon_index]),
                            "y_pred_normalized": float(normalized[sample_index, horizon_index]),
                            "scale": float(batch["scale"][sample_index, 0]),
                        }
                    )
    return rows


def run_rolling_experiment(
    data: ActualsData,
    *,
    stage: str,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    origins = get_rolling_origins(stage, data.dates)
    all_predictions: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    logs: list[str] = []
    for seed_value in config.get("seeds", [config.get("seed", 52)]):
        seed = int(seed_value)
        seed_everything(seed)
        warm_state: dict[str, torch.Tensor] | None = None
        for origin in origins:
            model, history, selected_epochs = _train_model_at_origin(
                data,
                origin,
                stage=stage,
                seed=seed,
                config=config,
                device=device,
                checkpoint_dir=Path(checkpoint_dir),
                warm_state=warm_state if bool(config.get("warm_start", False)) else None,
            )
            all_history.extend(history)
            predictions = _predict_origin(
                model,
                data,
                origin,
                stage=stage,
                seed=seed,
                config=config,
                device=device,
            )
            expected = len(data.part_ids) * int(config.get("prediction_length", 4))
            if len(predictions) != expected:
                raise AssertionError(f"Expected {expected} predictions, generated {len(predictions)}")
            all_predictions.extend(predictions)
            logs.append(
                f"stage={stage} seed={seed} block={origin.block} origin={origin.history_end:%Y-%m} "
                f"epochs={selected_epochs} predictions={len(predictions)}"
            )
            if bool(config.get("warm_start", False)):
                warm_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    prediction_frame = pd.DataFrame(all_predictions)
    key_columns = ["seed", "part_id", "target_date"]
    if prediction_frame.duplicated(key_columns).any():
        raise AssertionError("Rolling evaluation produced duplicate seed/part/month predictions")
    expected_months = {date for origin in origins for date in origin.target_dates}
    predicted_months = set(pd.DatetimeIndex(pd.to_datetime(prediction_frame["target_date"])))
    if predicted_months != expected_months:
        raise AssertionError("Rolling predictions do not exactly cover the scheduled 12 months")
    return prediction_frame, pd.DataFrame(all_history), logs
