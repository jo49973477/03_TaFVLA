"""Backward-compatible checkpoint and RNG helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from utils.common import resolved_config


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load current or pre-refactor checkpoints without requiring new fields."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Not a TaF-Adapter checkpoint: {path}")
    return checkpoint


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng: dict[str, Any] | None) -> None:
    if not rng:
        return
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def restore_rng(checkpoint: dict[str, Any]) -> None:
    restore_rng_state(checkpoint.get("rng"))


def save_checkpoint(
    path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, step: int, epoch: int, config: DictConfig | dict[str, Any],
    normalization: dict[str, torch.Tensor], train_keys: list[str], val_keys: list[str],
    wandb_state: dict[str, Any] | None = None,
    alignment_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": resolved_config(config),
        "normalization": {key: value.cpu() for key, value in normalization.items()},
        "split": {"train_groups": train_keys, "validation_groups": val_keys},
        "rng": capture_rng(),
    }
    if wandb_state:
        payload["wandb"] = dict(wandb_state)
    if alignment_state is not None:
        payload["alignment"] = dict(alignment_state)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

