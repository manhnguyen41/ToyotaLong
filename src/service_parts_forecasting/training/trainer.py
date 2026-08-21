from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpointing import load_model_checkpoint, save_checkpoint
from .losses import make_loss


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_loss: float
    history: list[dict[str, float | int | str]]


class Trainer:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.loss_fn = make_loss(str(config.get("loss", {}).get("name", "normalized_mae")))

    def _optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        options = self.config.get("optimizer", {})
        name = str(options.get("name", "adamw")).lower()
        kwargs = {
            "lr": float(options.get("learning_rate", 1e-3)),
            "weight_decay": float(options.get("weight_decay", 1e-4)),
        }
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), **kwargs)
        if name == "adam":
            return torch.optim.Adam(model.parameters(), **kwargs)
        raise ValueError(f"Unknown optimizer {name!r}")

    def _loader_loss(self, model: nn.Module, loader: DataLoader[dict[str, Any]]) -> float:
        model.eval()
        total_loss = 0.0
        total_items = 0
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target_normalized"].to(self.device)
                loss = self.loss_fn(model(inputs), targets)
                batch_size = inputs.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_items += batch_size
        return total_loss / max(total_items, 1)

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader[dict[str, Any]],
        *,
        epochs: int,
        checkpoint_path: str | Path,
        validation_loader: DataLoader[dict[str, Any]] | None = None,
        phase: str = "fit",
    ) -> FitResult:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        model.to(self.device)
        optimizer = self._optimizer(model)
        scheduler_config = self.config.get("scheduler", {})
        scheduler = None
        if str(scheduler_config.get("name", "none")).lower() == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=int(scheduler_config.get("patience", 5)),
                factor=float(scheduler_config.get("factor", 0.5)),
            )
        early = self.config.get("early_stopping", {})
        early_enabled = bool(early.get("enabled", True)) and validation_loader is not None
        patience = int(early.get("patience", 10))
        min_delta = float(early.get("min_delta", 0.0))
        clip_norm = float(self.config.get("gradient_clip_norm", 1.0))

        best_loss = np.inf
        best_epoch = 0
        stale_epochs = 0
        history: list[dict[str, float | int | str]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_items = 0
            for batch in train_loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target_normalized"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                batch_size = inputs.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_items += batch_size
            train_loss = total_loss / max(total_items, 1)
            validation_loss = (
                self._loader_loss(model, validation_loader) if validation_loader is not None else train_loss
            )
            if scheduler is not None:
                scheduler.step(validation_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])
            history.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": current_lr,
                }
            )
            if validation_loader is None:
                # Fixed-epoch final fits always use the requested last epoch.
                best_loss = validation_loss
                best_epoch = epoch
                save_checkpoint(checkpoint_path, model, epoch=epoch, metric=validation_loss)
            elif validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(checkpoint_path, model, epoch=epoch, metric=validation_loss)
            else:
                stale_epochs += 1
            if early_enabled and stale_epochs >= patience:
                break
        load_model_checkpoint(checkpoint_path, model, self.device)
        return FitResult(best_epoch=best_epoch, best_loss=float(best_loss), history=history)
