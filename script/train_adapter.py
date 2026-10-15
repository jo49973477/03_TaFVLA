"""Pre-train TaF-Adapter on temporally aligned TaF-Dataset episodes."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.taf_dataset import TaFDataset
from model.taf_adapter import TaFAdapter


METRIC_KEYS = (
    "loss_total", "loss_nce", "loss_quant", "loss_codebook", "loss_commitment",
    "loss_recon", "loss_recon_pressure", "loss_recon_force",
    "positive_cosine", "negative_cosine", "recall_at_1", "recall_at_5",
    "pressure_codebook_usage", "force_codebook_usage",
    "pressure_codebook_usage_fraction", "force_codebook_usage_fraction",
    "pressure_codebook_perplexity", "force_codebook_perplexity",
    "pressure_dead_codes", "force_dead_codes",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    for section in ("data", "force_encoder", "vq", "tactile_encoder", "loss", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Config is missing mapping section '{section}'")
    return config


def _episode_window_ranges(dataset: TaFDataset) -> list[range]:
    ranges = []
    previous = 0
    for end in dataset._ends:
        ranges.append(range(previous, end))
        previous = end
    return ranges


def split_by_group(
    dataset: TaFDataset, *, unit: str, validation_fraction: float, seed: int
) -> tuple[list[int], list[int], set[int], set[int], list[str], list[str]]:
    """Split whole sequences/episodes so overlapping windows cannot leak."""
    if unit not in {"sequence", "episode"}:
        raise ValueError("data.split_unit must be 'sequence' or 'episode'")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be between 0 and 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for episode_id, episode in enumerate(dataset.episodes):
        if unit == "sequence":
            key = episode.sequence_id
        else:
            key = f"{episode.sequence_id}/episode-{episode.episode_index}"
        groups[key].append(episode_id)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(
            f"Need at least two {unit} groups for a leakage-free train/validation split; got {len(keys)}"
        )
    random.Random(seed).shuffle(keys)
    val_count = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
    val_keys = set(keys[:val_count])
    train_keys = set(keys[val_count:])
    train_eps = {ep for key in train_keys for ep in groups[key]}
    val_eps = {ep for key in val_keys for ep in groups[key]}
    ranges = _episode_window_ranges(dataset)
    train_indices = [idx for ep in sorted(train_eps) for idx in ranges[ep]]
    val_indices = [idx for ep in sorted(val_eps) for idx in ranges[ep]]
    if not train_indices or not val_indices:
        raise ValueError("Train/validation split produced an empty window set")
    return train_indices, val_indices, train_eps, val_eps, sorted(train_keys), sorted(val_keys)


def audit_temporal_alignment(dataset: TaFDataset, windows_per_episode: int) -> None:
    """Decode representative windows after constructor-level all-frame timestamp checks."""
    if dataset.invalid_sequences:
        raise RuntimeError(
            "Dataset contains skipped invalid sequences; refusing to train: "
            + repr(dataset.invalid_sequences)
        )
    if windows_per_episode < 1:
        raise ValueError("alignment_decode_windows_per_episode must be >= 1")
    ranges = _episode_window_ranges(dataset)
    decoded = 0
    for episode_id, window_range in enumerate(ranges):
        count = len(window_range)
        offsets = np.linspace(0, count - 1, min(windows_per_episode, count), dtype=int)
        for offset in np.unique(offsets):
            sample = dataset[window_range.start + int(offset)]
            for key in ("tactile", "force", "pressure"):
                if not torch.isfinite(sample[key]).all():
                    raise ValueError(f"Non-finite {key} in episode index {episode_id}")
            meta = sample["metadata"]
            if not torch.all(torch.diff(meta["frame_indices"]) == 1):
                raise ValueError(f"Non-contiguous frames in episode index {episode_id}")
            if not torch.all(torch.diff(meta["timestamps"]) > 0):
                raise ValueError(f"Non-increasing sensor timestamps in episode index {episode_id}")
            if not torch.all(torch.diff(meta["video_timestamps"]) > 0):
                raise ValueError(f"Non-increasing video timestamps in episode index {episode_id}")
            decoded += 1
    print(f"Temporal alignment audit: PASS ({len(dataset.episodes)} episodes, {decoded} decoded windows)")


def compute_normalization(
    dataset: TaFDataset, train_episode_ids: set[int], chunk_frames: int, eps: float
) -> dict[str, torch.Tensor]:
    """Compute per-taxel/per-wrench-component population stats from train frames only."""
    if chunk_frames < 1 or eps <= 0:
        raise ValueError("normalization_chunk_frames and normalization_eps must be positive")
    p_sum = torch.zeros(12, 12, dtype=torch.float64)
    p_sq = torch.zeros_like(p_sum)
    f_sum = torch.zeros(6, dtype=torch.float64)
    f_sq = torch.zeros_like(f_sum)
    count = 0
    for episode_id in sorted(train_episode_ids):
        episode = dataset.episodes[episode_id]
        for start in range(0, episode.length, chunk_frames):
            offsets = np.arange(start, min(start + chunk_frames, episode.length), dtype=np.int64)
            force_np, pressure_np = dataset._sensor_rows(episode, offsets)
            pressure = torch.from_numpy(pressure_np).double()
            force = torch.from_numpy(force_np).double()
            if not torch.isfinite(pressure).all() or not torch.isfinite(force).all():
                raise ValueError(f"Non-finite force data in training episode {episode.sequence_id}/{episode.episode_index}")
            p_sum += pressure.sum(0)
            p_sq += pressure.square().sum(0)
            f_sum += force.sum(0)
            f_sq += force.square().sum(0)
            count += len(offsets)
    if count < 2:
        raise ValueError("At least two training frames are required for normalization")
    p_mean = p_sum / count
    f_mean = f_sum / count
    p_std = (p_sq / count - p_mean.square()).clamp_min(0).sqrt().clamp_min(eps)
    f_std = (f_sq / count - f_mean.square()).clamp_min(0).sqrt().clamp_min(eps)
    print(f"Normalization stats: {count} unique training frames (validation excluded)")
    return {
        "pressure_mean": p_mean.float(), "pressure_std": p_std.float(),
        "force_mean": f_mean.float(), "force_std": f_std.float(),
    }


class TemporalDiverseBatchSampler(Sampler[list[int]]):
    """Greedily separate nearby windows from the same episode within a batch.

    If the remaining pool makes the requested gap impossible, it still emits a
    batch (and therefore samples every window) rather than silently discarding data.
    """

    def __init__(
        self, dataset: TaFDataset, subset_indices: list[int], batch_size: int,
        min_gap: int, seed: int,
    ) -> None:
        if batch_size < 2:
            raise ValueError("InfoNCE training requires batch_size >= 2")
        self.batch_size = batch_size
        self.min_gap = max(0, min_gap)
        self.seed = seed
        self.epoch = 0
        ep_ids = {id(ep): i for i, ep in enumerate(dataset.episodes)}
        self.items: list[tuple[int, int, int]] = []
        for local_index, global_index in enumerate(subset_indices):
            episode, start = dataset._locate(global_index)
            self.items.append((local_index, ep_ids[id(episode)], start))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.items) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        items = list(self.items)
        random.Random(self.seed + self.epoch).shuffle(items)
        queue = deque(items)
        while queue:
            selected: list[tuple[int, int, int]] = []
            deferred: list[tuple[int, int, int]] = []
            for _ in range(len(queue)):
                item = queue.popleft()
                compatible = all(
                    item[1] != other[1] or abs(item[2] - other[2]) >= self.min_gap
                    for other in selected
                )
                if compatible and len(selected) < self.batch_size:
                    selected.append(item)
                else:
                    deferred.append(item)
                if len(selected) == self.batch_size:
                    break
            queue.extend(deferred)
            if len(selected) == 1 and queue:
                selected.append(queue.popleft())
            if len(selected) >= 2:
                yield [item[0] for item in selected]


def build_model(config: dict[str, Any]) -> TaFAdapter:
    data_cfg = config["data"]
    tactile_cfg = dict(config["tactile_encoder"])
    tactile_cfg["max_window_size"] = max(
        int(tactile_cfg.get("max_window_size", data_cfg["window_size"])),
        int(data_cfg["window_size"]),
    )
    return TaFAdapter(
        window_size=int(data_cfg["window_size"]),
        force_encoder=config["force_encoder"],
        vq=config["vq"],
        tactile_encoder=tactile_cfg,
        loss=config["loss"],
        normalization_eps=float(data_cfg.get("normalization_eps", 1e-6)),
    )


def scalar_metrics(outputs: dict[str, Any]) -> dict[str, float]:
    return {key: float(outputs[key].detach()) for key in METRIC_KEYS}


@torch.no_grad()
def validate(
    model: TaFAdapter, loader: DataLoader, device: torch.device,
    amp_enabled: bool, max_batches: int | None,
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


def save_checkpoint(
    path: Path, *, model: TaFAdapter, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, step: int, epoch: int, config: dict[str, Any],
    normalization: dict[str, torch.Tensor], train_keys: list[str], val_keys: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "step": step, "epoch": epoch,
        "config": config,
        "normalization": {key: value.cpu() for key, value in normalization.items()},
        "split": {"train_groups": train_keys, "validation_groups": val_keys},
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng(checkpoint: dict[str, Any]) -> None:
    rng = checkpoint.get("rng")
    if not rng:
        return
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/taf_adapter.yaml"))
    parser.add_argument("--root-dir", type=Path, help="override data.root_dir")
    parser.add_argument("--output-dir", type=Path, help="override training.output_dir")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.root_dir is not None:
        config["data"]["root_dir"] = str(args.root_dir)
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir)
    if config["data"].get("root_dir") is None:
        parser.error("dataset root is required via --root-dir or data.root_dir")
    train_cfg, data_cfg = config["training"], config["data"]
    seed = int(train_cfg["seed"])
    set_seed(seed)

    dataset = TaFDataset(
        data_cfg["root_dir"], window_size=int(data_cfg["window_size"]),
        stride=int(data_cfg["stride"]), image_size=tuple(data_cfg["image_size"]),
        sensor_types=data_cfg.get("sensor_types"), max_sequences=data_cfg.get("max_sequences"),
        skip_invalid_sequences=bool(data_cfg.get("skip_invalid_sequences", False)),
    )
    audit_temporal_alignment(dataset, int(data_cfg.get("alignment_decode_windows_per_episode", 1)))
    split = split_by_group(
        dataset, unit=str(data_cfg.get("split_unit", "sequence")),
        validation_fraction=float(data_cfg.get("validation_fraction", 0.1)),
        seed=int(data_cfg["split_seed"]),
    )
    train_indices, val_indices, train_eps, _, train_keys, val_keys = split
    print(
        f"Split by {data_cfg.get('split_unit', 'sequence')}: "
        f"train={len(train_indices)} windows/{len(train_keys)} groups, "
        f"validation={len(val_indices)} windows/{len(val_keys)} groups"
    )

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_split = checkpoint.get("split", {})
        if saved_split.get("train_groups") != train_keys or saved_split.get("validation_groups") != val_keys:
            raise ValueError("Current dataset split does not match the resume checkpoint")
        normalization = checkpoint["normalization"]
    else:
        normalization = compute_normalization(
            dataset, train_eps, int(data_cfg.get("normalization_chunk_frames", 8192)),
            float(data_cfg.get("normalization_eps", 1e-6)),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.set_normalization(**normalization)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]),
    )
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    step, epoch = 0, 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        step, epoch = int(checkpoint["step"]), int(checkpoint.get("epoch", 0))
        restore_rng(checkpoint)
        print(f"Resumed {args.resume} at optimizer step {step}")

    batch_sampler = TemporalDiverseBatchSampler(
        dataset, train_indices, batch_size=int(train_cfg["batch_size"]),
        min_gap=int(train_cfg.get("min_same_episode_window_gap", data_cfg["window_size"])),
        seed=seed,
    )
    workers = int(train_cfg["num_workers"])
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_sampler=batch_sampler, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=int(train_cfg["batch_size"]), shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )

    output_dir = Path(train_cfg.get("output_dir", "outputs/taf_adapter"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    max_steps = int(train_cfg["max_steps"])
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    if max_steps < 1 or accumulation < 1:
        raise ValueError("max_steps and gradient_accumulation_steps must be >= 1")
    optimizer.zero_grad(set_to_none=True)
    rolling: dict[str, list[float]] = defaultdict(list)
    micro_step = step * accumulation
    started = time.time()

    while step < max_steps:
        batch_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            tactile = batch["tactile"].to(device, non_blocking=True)
            force = batch["force"].to(device, non_blocking=True)
            pressure = batch["pressure"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(tactile=tactile, force=force, pressure=pressure)
                scaled_loss = outputs["loss_total"] / accumulation
            scaler.scale(scaled_loss).backward()
            for key, value in scalar_metrics(outputs).items():
                rolling[key].append(value)
            micro_step += 1
            if micro_step % accumulation:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                float(train_cfg.get("grad_clip_norm", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % int(train_cfg["log_every"]) == 0:
                record = {
                    "step": step, "epoch": epoch, "seconds": time.time() - started,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                    **{f"train/{key}": sum(values) / len(values) for key, values in rolling.items()},
                }
                print(json.dumps(record, sort_keys=True))
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                rolling.clear()

            validate_every = int(train_cfg.get("validate_every", 0))
            if validate_every > 0 and step % validate_every == 0:
                val_metrics = validate(
                    model, val_loader, device, amp_enabled,
                    int(train_cfg["validation_batches"]) if train_cfg.get("validation_batches") else None,
                )
                record = {"step": step, "epoch": epoch, **val_metrics}
                print(json.dumps(record, sort_keys=True))
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                model.train()

            if step % int(train_cfg["save_every"]) == 0:
                save_checkpoint(
                    output_dir / f"step_{step:08d}.pt", model=model, optimizer=optimizer,
                    scaler=scaler, step=step, epoch=epoch, config=config,
                    normalization=normalization, train_keys=train_keys, val_keys=val_keys,
                )
            if step >= max_steps:
                break
        epoch += 1

    save_checkpoint(
        output_dir / "last.pt", model=model, optimizer=optimizer, scaler=scaler,
        step=step, epoch=epoch, config=config, normalization=normalization,
        train_keys=train_keys, val_keys=val_keys,
    )
    print(f"Training complete at step {step}; checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
