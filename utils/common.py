"""Configuration, reproducibility, and path helpers."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf


_CONFIG_KEYS = {
    "data": {
        "root_dir", "window_size", "stride", "image_size", "max_sequences",
        "sensor_types", "split_seed", "split_unit", "validation_fraction",
        "skip_invalid_sequences", "alignment_decode_windows_per_episode",
        "normalization_eps", "normalization_chunk_frames",
    },
    "force_encoder": {
        "hidden_dim", "pressure_latent_dim", "force_latent_dim", "mlp_layers",
    },
    "vq": {"pressure_codebook_size", "force_codebook_size", "commitment_beta"},
    "tactile_encoder": {
        "backbone", "pretrained", "freeze_backbone", "transformer_dim",
        "transformer_layers", "num_heads", "dropout", "max_window_size",
    },
    "loss": {"lambda_quant", "lambda_recon", "temperature", "nce_mode"},
    "training": {
        "batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "max_steps", "num_workers", "seed", "amp", "grad_clip_norm", "log_every",
        "validate_every", "validation_batches", "save_every", "output_dir", "resume",
        "min_same_episode_window_gap",
    },
    "wandb": {
        "enabled", "project", "entity", "name", "group", "tags", "mode",
        "log_every", "watch_model",
    },
    "check": {"overfit_steps", "skip_overfit", "no_pretrained", "window_sizes"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path) -> Path:
    """Resolve a user path relative to Hydra's original launch directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path(HydraConfig.get().runtime.cwd) if HydraConfig.initialized() else Path.cwd()
    return (base / path).resolve()


def resolved_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
    else:
        value = dict(config)
    if not isinstance(value, dict):
        raise TypeError("Config must resolve to a mapping")
    return value


def _mapping(config: DictConfig, section: str) -> DictConfig:
    value = config.get(section)
    if not isinstance(value, DictConfig):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None, inclusive_maximum: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None:
        invalid = number > maximum if inclusive_maximum else number >= maximum
        if invalid:
            operator = "<=" if inclusive_maximum else "<"
            raise ValueError(f"{name} must be {operator} {maximum}, got {value}")
    return number


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _boolean(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")


def validate_config(config: DictConfig, *, require_data_root: bool = False) -> None:
    """Validate known keys plus the high-impact types and ranges."""
    unknown_sections = set(config.keys()) - set(_CONFIG_KEYS)
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}")
    missing_sections = set(_CONFIG_KEYS) - set(config.keys())
    if missing_sections:
        raise ValueError(f"Missing config section(s): {sorted(missing_sections)}")
    sections = {name: _mapping(config, name) for name in _CONFIG_KEYS}
    for name, allowed in _CONFIG_KEYS.items():
        unknown = set(sections[name].keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown config key(s) in {name}: {sorted(unknown)}")

    data = sections["data"]
    if require_data_root and not data.root_dir:
        raise ValueError("data.root_dir is required for training")
    if data.root_dir is not None and not isinstance(data.root_dir, str):
        raise TypeError("data.root_dir must be null or a path string")
    _integer(data.window_size, "data.window_size", minimum=1)
    _integer(data.stride, "data.stride", minimum=1)
    if not isinstance(data.image_size, (list, tuple, ListConfig)) or len(data.image_size) != 2:
        raise TypeError("data.image_size must contain [height, width]")
    for index, size in enumerate(data.image_size):
        _integer(size, f"data.image_size[{index}]", minimum=1)
    if data.max_sequences is not None:
        _integer(data.max_sequences, "data.max_sequences", minimum=1)
    if data.sensor_types is not None:
        if isinstance(data.sensor_types, str):
            pass
        elif not isinstance(data.sensor_types, (list, tuple, ListConfig)) or not all(
            isinstance(value, str) for value in data.sensor_types
        ):
            raise TypeError("data.sensor_types must be null, a string, or a list of strings")
    _integer(data.split_seed, "data.split_seed", minimum=0)
    if data.split_unit not in {"sequence", "episode"}:
        raise ValueError("data.split_unit must be 'sequence' or 'episode'")
    validation_fraction = _number(data.validation_fraction, "data.validation_fraction")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be between 0 and 1")
    _boolean(data.skip_invalid_sequences, "data.skip_invalid_sequences")
    _integer(
        data.alignment_decode_windows_per_episode,
        "data.alignment_decode_windows_per_episode", minimum=1,
    )
    if _number(data.normalization_eps, "data.normalization_eps") <= 0:
        raise ValueError("data.normalization_eps must be positive")
    _integer(data.normalization_chunk_frames, "data.normalization_chunk_frames", minimum=1)

    force = sections["force_encoder"]
    for key in ("hidden_dim", "pressure_latent_dim", "force_latent_dim", "mlp_layers"):
        _integer(force[key], f"force_encoder.{key}", minimum=1)

    vq = sections["vq"]
    _integer(vq.pressure_codebook_size, "vq.pressure_codebook_size", minimum=1)
    _integer(vq.force_codebook_size, "vq.force_codebook_size", minimum=1)
    _number(vq.commitment_beta, "vq.commitment_beta", minimum=0.0)

    tactile = sections["tactile_encoder"]
    if not isinstance(tactile.backbone, str) or not tactile.backbone:
        raise TypeError("tactile_encoder.backbone must be a non-empty string")
    _boolean(tactile.pretrained, "tactile_encoder.pretrained")
    _boolean(tactile.freeze_backbone, "tactile_encoder.freeze_backbone")
    transformer_dim = _integer(tactile.transformer_dim, "tactile_encoder.transformer_dim", minimum=1)
    _integer(tactile.transformer_layers, "tactile_encoder.transformer_layers", minimum=1)
    num_heads = _integer(tactile.num_heads, "tactile_encoder.num_heads", minimum=1)
    if transformer_dim % num_heads:
        raise ValueError("tactile_encoder.transformer_dim must be divisible by num_heads")
    _number(tactile.dropout, "tactile_encoder.dropout", minimum=0.0,
            maximum=1.0, inclusive_maximum=False)
    _integer(tactile.max_window_size, "tactile_encoder.max_window_size", minimum=1)

    loss = sections["loss"]
    _number(loss.lambda_quant, "loss.lambda_quant", minimum=0.0)
    _number(loss.lambda_recon, "loss.lambda_recon", minimum=0.0)
    if _number(loss.temperature, "loss.temperature") <= 0:
        raise ValueError("loss.temperature must be positive")
    if loss.nce_mode not in {"paper", "symmetric"}:
        raise ValueError("loss.nce_mode must be 'paper' or 'symmetric'")

    training = sections["training"]
    _integer(training.batch_size, "training.batch_size", minimum=2)
    _integer(training.gradient_accumulation_steps, "training.gradient_accumulation_steps", minimum=1)
    if _number(training.learning_rate, "training.learning_rate") <= 0:
        raise ValueError("training.learning_rate must be positive")
    _number(training.weight_decay, "training.weight_decay", minimum=0.0)
    _integer(training.max_steps, "training.max_steps", minimum=1)
    _integer(training.num_workers, "training.num_workers", minimum=0)
    _integer(training.seed, "training.seed", minimum=0)
    _boolean(training.amp, "training.amp")
    if _number(training.grad_clip_norm, "training.grad_clip_norm") <= 0:
        raise ValueError("training.grad_clip_norm must be positive")
    _integer(training.log_every, "training.log_every", minimum=1)
    _integer(training.validate_every, "training.validate_every", minimum=0)
    if training.validation_batches is not None:
        _integer(training.validation_batches, "training.validation_batches", minimum=1)
    _integer(training.save_every, "training.save_every", minimum=1)
    if not isinstance(training.output_dir, str) or not training.output_dir:
        raise TypeError("training.output_dir must be a non-empty string")
    if training.resume is not None and not isinstance(training.resume, str):
        raise TypeError("training.resume must be null or a path string")
    _integer(
        training.min_same_episode_window_gap,
        "training.min_same_episode_window_gap", minimum=0,
    )

    wandb = sections["wandb"]
    _boolean(wandb.enabled, "wandb.enabled")
    if not isinstance(wandb.project, str) or not wandb.project:
        raise TypeError("wandb.project must be a non-empty string")
    for key in ("entity", "name", "group"):
        if wandb[key] is not None and not isinstance(wandb[key], str):
            raise TypeError(f"wandb.{key} must be null or a string")
    if not isinstance(wandb.tags, (list, tuple, ListConfig)) or not all(
        isinstance(tag, str) for tag in wandb.tags
    ):
        raise TypeError("wandb.tags must be a list of strings")
    if wandb.mode not in {"online", "offline"}:
        raise ValueError("wandb.mode must be 'online' or 'offline'")
    _integer(wandb.log_every, "wandb.log_every", minimum=1)
    _boolean(wandb.watch_model, "wandb.watch_model")

    check = sections["check"]
    _integer(check.overfit_steps, "check.overfit_steps", minimum=1)
    _boolean(check.skip_overfit, "check.skip_overfit")
    _boolean(check.no_pretrained, "check.no_pretrained")
    if not isinstance(check.window_sizes, (list, tuple, ListConfig)) or not check.window_sizes:
        raise TypeError("check.window_sizes must be a non-empty list")
    for index, size in enumerate(check.window_sizes):
        _integer(size, f"check.window_sizes[{index}]", minimum=1)

    OmegaConf.set_struct(config, True)

