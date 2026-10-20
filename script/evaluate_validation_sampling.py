"""Compare sequential and temporally diverse validation without training."""

from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from dataset.taf_dataset import TaFDataset
from script.train_adapter import validate
from utils.checkpoint import capture_rng, load_checkpoint, restore_rng_state
from utils.common import resolve_path, set_seed, validate_config
from utils.config import register_config_schema
from utils.data import FixedValidationBatchSampler, split_by_group
from utils.model_factory import build_model


register_config_schema()


def _limited(value: int, maximum: int | None) -> int:
    return value if maximum is None else min(value, maximum)


@hydra.main(version_base="1.3", config_path="../config", config_name="taf_adapter")
def main(cfg: DictConfig) -> None:
    """Evaluate one checkpoint under both validation negative protocols."""
    validate_config(cfg, require_data_root=True)
    if not cfg.training.resume:
        raise ValueError("training.resume must point to the checkpoint to evaluate")

    set_seed(int(cfg.training.seed))
    checkpoint_path = resolve_path(cfg.training.resume)
    checkpoint = load_checkpoint(checkpoint_path)
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint has no resolved training config")
    runtime_cfg = cfg
    cfg = OmegaConf.merge(runtime_cfg, OmegaConf.create(saved_config))
    # Model/data semantics come from the checkpoint. Only evaluation runtime
    # paths, worker count, batch limit, and AMP choice remain CLI-configurable.
    cfg.data.root_dir = runtime_cfg.data.root_dir
    cfg.data.timestamp_cache_enabled = runtime_cfg.data.timestamp_cache_enabled
    cfg.data.timestamp_cache_dir = runtime_cfg.data.timestamp_cache_dir
    cfg.data.show_initialization_progress = runtime_cfg.data.show_initialization_progress
    cfg.training.resume = runtime_cfg.training.resume
    cfg.training.output_dir = runtime_cfg.training.output_dir
    cfg.training.num_workers = runtime_cfg.training.num_workers
    cfg.training.validation_batches = runtime_cfg.training.validation_batches
    cfg.training.amp = runtime_cfg.training.amp
    validate_config(cfg, require_data_root=True)
    data_cfg, train_cfg = cfg.data, cfg.training
    dataset = TaFDataset(
        resolve_path(data_cfg.root_dir),
        window_size=int(data_cfg.window_size),
        stride=int(data_cfg.stride),
        image_size=tuple(data_cfg.image_size),
        sensor_types=data_cfg.sensor_types,
        max_sequences=data_cfg.max_sequences,
        skip_invalid_sequences=bool(data_cfg.skip_invalid_sequences),
        timestamp_cache_enabled=bool(data_cfg.timestamp_cache_enabled),
        timestamp_cache_dir=(
            resolve_path(data_cfg.timestamp_cache_dir)
            if data_cfg.timestamp_cache_dir is not None else None
        ),
        show_initialization_progress=bool(data_cfg.show_initialization_progress),
    )
    train_indices, val_indices, _, _, train_keys, val_keys = split_by_group(
        dataset,
        unit=str(data_cfg.split_unit),
        validation_fraction=float(data_cfg.validation_fraction),
        seed=int(data_cfg.split_seed),
    )
    saved_split = checkpoint.get("split", {})
    if (
        saved_split.get("train_groups") != train_keys
        or saved_split.get("validation_groups") != val_keys
    ):
        raise ValueError("Current dataset split does not match the checkpoint")
    if "normalization" not in checkpoint:
        raise ValueError("Checkpoint has no normalization statistics")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The checkpoint immediately supplies every weight, so evaluation must not
    # download or initialize a pretrained backbone first.
    model = build_model(cfg, pretrained=False)
    model.set_normalization(**checkpoint["normalization"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    workers = int(train_cfg.num_workers)
    batch_size = int(train_cfg.batch_size)
    maximum_batches = (
        int(train_cfg.validation_batches)
        if train_cfg.validation_batches is not None else None
    )

    old_generator = torch.Generator().manual_seed(int(data_cfg.split_seed))
    sequential_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=old_generator,
    )
    validation_gap = max(
        int(data_cfg.window_size), int(train_cfg.min_same_episode_window_gap),
    )
    diverse_sampler = FixedValidationBatchSampler(
        dataset,
        val_indices,
        batch_size=batch_size,
        min_gap=validation_gap,
        seed=int(data_cfg.split_seed),
    )
    new_generator = torch.Generator().manual_seed(int(data_cfg.split_seed))
    diverse_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_sampler=diverse_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=new_generator,
    )

    rng = capture_rng()
    try:
        sequential_metrics = validate(
            model, sequential_loader, device,
            bool(train_cfg.amp) and device.type == "cuda", maximum_batches,
        )
        restore_rng_state(rng)
        diverse_metrics = validate(
            model, diverse_loader, device,
            bool(train_cfg.amp) and device.type == "cuda", maximum_batches,
        )
    finally:
        restore_rng_state(rng)

    sequential_batch_count = _limited(
        math.ceil(len(val_indices) / batch_size), maximum_batches,
    )
    diverse_batch_count = _limited(len(diverse_sampler), maximum_batches)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "training_performed": False,
        "sequential_validation": {
            "protocol": "legacy contiguous windows, shuffle=false",
            "batch_count": sequential_batch_count,
            "sample_count": min(len(val_indices), sequential_batch_count * batch_size),
            "may_include_partial_final_batch": (
                maximum_batches is None
                or maximum_batches >= math.ceil(len(val_indices) / batch_size)
            ) and len(val_indices) % batch_size != 0,
            "metrics": sequential_metrics,
        },
        "temporally_diverse_validation": {
            "protocol": "fixed full batches with strict same-episode start gap",
            "batch_count": diverse_batch_count,
            "sample_count": diverse_batch_count * batch_size,
            "sampling": diverse_sampler.report.as_dict(),
            "metrics": diverse_metrics,
        },
    }
    output_path = resolve_path(train_cfg.output_dir) / "validation_sampling_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()
