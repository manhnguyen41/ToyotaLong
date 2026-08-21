from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config with simple relative ``defaults`` inheritance."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    defaults = raw.pop("defaults", [])
    if isinstance(defaults, str):
        defaults = [defaults]
    merged: dict[str, Any] = {}
    for default in defaults:
        merged = _deep_merge(merged, load_config(config_path.parent / default))
    return _deep_merge(merged, raw)


def resolve_runtime_config(
    config: dict[str, Any], *, smoke_test: bool = False
) -> dict[str, Any]:
    resolved = deepcopy(config)
    if smoke_test:
        resolved["smoke_test"] = True
        resolved["epochs"] = 1
        resolved["final_epochs"] = 1
        resolved["num_workers"] = 0
        resolved["batch_size"] = min(int(resolved.get("batch_size", 64)), 64)
        resolved["max_parts"] = min(int(resolved.get("max_parts", 32)), 32)
        resolved["seeds"] = [int(resolved.get("seed", 52))]
        if isinstance(resolved.get("model"), dict):
            resolved["model"]["hidden_size"] = min(
                int(resolved["model"].get("hidden_size", 16)), 16
            )
            resolved["model"]["num_layers"] = 1
    resolved.setdefault("seeds", [int(resolved.get("seed", 52))])
    return resolved


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
