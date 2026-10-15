"""Pre-train TaF-Adapter on temporally aligned TaF-Dataset episodes."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from dataset.taf_dataset import TaFDataset
from utils.checkpoint import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    restore_rng_state,
    save_checkpoint,
)
from utils.common import resolve_path, resolved_config, set_seed, validate_config
from utils.config import register_config_schema
from utils.data import (
    TemporalDiverseBatchSampler,
    audit_temporal_alignment,
    compute_normalization,
    split_by_group,
)
from utils.logging import ExperimentLogger, scalar_metrics
from utils.model_factory import build_model


register_config_schema()


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    batches = 0
    for batch in loader:
        tactile = batch["tactile"].to(device, non_blocking=True)
        force = batch["force"].to(device, non_blocking=True)
        pressure = batch["pressure"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(tactile=tactile, force=force, pressure=pressure)
        for key, value in scalar_metrics(outputs).items():
            sums[key] += value
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    if batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {f"val/{key}": value / batches for key, value in sums.items()}


def _mean_metrics(values: dict[str, list[float]], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": sum(items) / len(items) for key, items in values.items()}


def run_training(cfg: DictConfig) -> None:
    """Run training from a fully composed and validated Hydra config."""
    validate_config(cfg, require_data_root=True)
    train_cfg, data_cfg = cfg.training, cfg.data
    seed = int(train_cfg.seed)
    set_seed(seed)

    dataset = TaFDataset(
        resolve_path(data_cfg.root_dir),
        window_size=int(data_cfg.window_size),
        stride=int(data_cfg.stride),
        image_size=tuple(data_cfg.image_size),
        sensor_types=data_cfg.sensor_types,
        max_sequences=data_cfg.max_sequences,
        skip_invalid_sequences=bool(data_cfg.skip_invalid_sequences),
    )
    audit_temporal_alignment(dataset, int(data_cfg.alignment_decode_windows_per_episode))
    split = split_by_group(
        dataset,
        unit=str(data_cfg.split_unit),
        validation_fraction=float(data_cfg.validation_fraction),
        seed=int(data_cfg.split_seed),
    )
    train_indices, val_indices, train_eps, _, train_keys, val_keys = split
    print(
        f"Split by {data_cfg.split_unit}: "
        f"train={len(train_indices)} windows/{len(train_keys)} groups, "
        f"validation={len(val_indices)} windows/{len(val_keys)} groups"
    )

    resume_path = resolve_path(train_cfg.resume) if train_cfg.resume else None
    checkpoint = load_checkpoint(resume_path) if resume_path else None
    if checkpoint is not None:
        saved_split = checkpoint.get("split", {})
        if (
            saved_split.get("train_groups") != train_keys
            or saved_split.get("validation_groups") != val_keys
        ):
            raise ValueError("Current dataset split does not match the resume checkpoint")
        if "normalization" not in checkpoint:
            raise ValueError("Resume checkpoint has no normalization statistics")
        normalization = checkpoint["normalization"]
    else:
        normalization = compute_normalization(
            dataset,
            train_eps,
            int(data_cfg.normalization_chunk_frames),
            float(data_cfg.normalization_eps),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model.set_normalization(**normalization)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.weight_decay),
    )
    amp_enabled = bool(train_cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    step, epoch = 0, 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        step, epoch = int(checkpoint["step"]), int(checkpoint.get("epoch", 0))
        restore_rng(checkpoint)
        print(f"Resumed {resume_path} at optimizer step {step}")

    batch_sampler = TemporalDiverseBatchSampler(
        dataset,
        train_indices,
        batch_size=int(train_cfg.batch_size),
        min_gap=int(train_cfg.min_same_episode_window_gap),
        seed=seed,
    )
    workers = int(train_cfg.num_workers)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    output_dir = resolve_path(train_cfg.output_dir)
    config_dict = resolved_config(cfg)
    logger = ExperimentLogger(
        output_dir,
        cfg.wandb,
        checkpoint.get("wandb") if checkpoint is not None else None,
    )
    rng_before_wandb = capture_rng()
    try:
        logger.start(cfg, model)
    finally:
        # Experiment tracking must not change training randomness.
        restore_rng_state(rng_before_wandb)
    max_steps = int(train_cfg.max_steps)
    accumulation = int(train_cfg.gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    local_rolling: dict[str, list[float]] = defaultdict(list)
    wandb_rolling: dict[str, list[float]] = defaultdict(list)
    micro_step = step * accumulation
    started = time.time()

    failed = True
    try:
        while step < max_steps:
            batch_sampler.set_epoch(epoch)
            model.train()
            for batch in train_loader:
                tactile = batch["tactile"].to(device, non_blocking=True)
                force = batch["force"].to(device, non_blocking=True)
                pressure = batch["pressure"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    outputs = model(tactile=tactile, force=force, pressure=pressure)
                    scaled_loss = outputs["loss_total"] / accumulation
                scaler.scale(scaled_loss).backward()
                batch_metrics = scalar_metrics(outputs)
                for key, value in batch_metrics.items():
                    local_rolling[key].append(value)
                    if cfg.wandb.enabled:
                        wandb_rolling[key].append(value)
                micro_step += 1
                if micro_step % accumulation:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    float(train_cfg.grad_clip_norm),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                elapsed = time.time() - started
                learning_rate = float(optimizer.param_groups[0]["lr"])
                optimization = {
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": float(grad_norm),
                    "global_step": step,
                    "epoch": epoch,
                    "train/elapsed_time_seconds": elapsed,
                }

                if step % int(train_cfg.log_every) == 0:
                    local_record: dict[str, Any] = {
                        "step": step,
                        "epoch": epoch,
                        "seconds": elapsed,
                        "learning_rate": learning_rate,
                        "grad_norm": float(grad_norm),
                        **_mean_metrics(local_rolling, "train"),
                    }
                    logger.write(local_record)
                    local_rolling.clear()

                wandb_payload: dict[str, float | int] = {}
                if cfg.wandb.enabled and step % int(cfg.wandb.log_every) == 0:
                    wandb_payload.update(_mean_metrics(wandb_rolling, "train"))
                    wandb_payload.update(optimization)
                    wandb_rolling.clear()

                validate_every = int(train_cfg.validate_every)
                if validate_every > 0 and step % validate_every == 0:
                    val_metrics = validate(
                        model,
                        val_loader,
                        device,
                        amp_enabled,
                        int(train_cfg.validation_batches)
                        if train_cfg.validation_batches is not None
                        else None,
                    )
                    logger.write({"step": step, "epoch": epoch, **val_metrics})
                    wandb_payload.update(val_metrics)
                    wandb_payload.update(optimization)
                    model.train()

                if wandb_payload:
                    logger.log_wandb(wandb_payload, step)

                if step % int(train_cfg.save_every) == 0:
                    save_checkpoint(
                        output_dir / f"step_{step:08d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        epoch=epoch,
                        config=config_dict,
                        normalization=normalization,
                        train_keys=train_keys,
                        val_keys=val_keys,
                        wandb_state=logger.wandb_state(),
                    )
                if step >= max_steps:
                    break
            epoch += 1

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            epoch=epoch,
            config=config_dict,
            normalization=normalization,
            train_keys=train_keys,
            val_keys=val_keys,
            wandb_state=logger.wandb_state(),
        )
        print(f"Training complete at step {step}; checkpoint: {output_dir / 'last.pt'}")
        failed = False
    finally:
        logger.finish(exit_code=1 if failed else 0)


@hydra.main(version_base="1.3", config_path="../config", config_name="taf_adapter")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
