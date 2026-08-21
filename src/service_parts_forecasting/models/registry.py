from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .base import BaseForecastModel

ModelType = TypeVar("ModelType", bound=type[BaseForecastModel])
_MODELS: dict[str, type[BaseForecastModel]] = {}


def register_model(name: str) -> Callable[[ModelType], ModelType]:
    def decorator(model_class: ModelType) -> ModelType:
        if name in _MODELS:
            raise ValueError(f"Model {name!r} is already registered")
        _MODELS[name] = model_class
        return model_class

    return decorator


def create_model(config: dict[str, Any], *, input_size: int) -> BaseForecastModel:
    name = str(config.get("name", ""))
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}; registered models: {sorted(_MODELS)}")
    kwargs = {key: value for key, value in config.items() if key not in {"name", "input_size"}}
    return _MODELS[name](input_size=input_size, **kwargs)


def registered_models() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))

