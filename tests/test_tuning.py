from __future__ import annotations

from typing import Any

from service_parts_forecasting.config import load_config
from service_parts_forecasting.tuning import sample_search_space


class FixedTrial:
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return choices[-1]

    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int:
        return high

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        return high


def test_tuning_config_and_nested_search_space() -> None:
    config = load_config("configs/tuning/lstm_optuna.yaml")
    tuning = config.pop("tuning")
    sampled = sample_search_space(FixedTrial(), config, tuning["search_space"])
    assert sampled["context_length"] == 24
    assert sampled["model"]["hidden_size"] == 128
    assert sampled["model"]["num_layers"] == 2
    assert sampled["model"]["dropout"] == 0.2
    assert sampled["optimizer"]["learning_rate"] == 0.003
    assert sampled["optimizer"]["weight_decay"] == 0.001
    assert sampled["batch_size"] == 512
    assert config["model"]["hidden_size"] == 64
