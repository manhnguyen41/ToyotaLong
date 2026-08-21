from __future__ import annotations

from typing import Any

from service_parts_forecasting.config import load_config
from service_parts_forecasting.tuning import import_completed_trials, sample_search_space


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


def test_completed_trials_can_be_imported_idempotently(tmp_path) -> None:
    config = load_config("configs/tuning/lstm_optuna.yaml")
    results = tmp_path / "tuning_results.csv"
    results.write_text(
        "trial,state,value,duration_seconds,param_context_length,param_model.hidden_size,"
        "param_model.num_layers,param_model.dropout,param_optimizer.learning_rate,"
        "param_optimizer.weight_decay,param_batch_size,attr_suggested_final_epochs\n"
        "0,COMPLETE,0.42,10,12,64,2,0.1,0.001,0.0001,256,17\n",
        encoding="utf-8",
    )
    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    first = import_completed_trials(results, config, storage=storage)
    second = import_completed_trials(results, config, storage=storage)
    assert first["imported"] == 1
    assert first["total_trials_in_study"] == 1
    assert first["best_value"] == 0.42
    assert second["imported"] == 0
    assert second["total_trials_in_study"] == 1
