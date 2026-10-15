"""Reusable dataset splitting, auditing, normalization, and sampling."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler

from dataset.taf_dataset import TaFDataset


def episode_window_ranges(dataset: TaFDataset) -> list[range]:
    ranges = []
    previous = 0
    for end in dataset._ends:
        ranges.append(range(previous, end))
        previous = end
    return ranges


def split_by_group(
    dataset: TaFDataset, *, unit: str, validation_fraction: float, seed: int,
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
    train_eps = {episode for key in train_keys for episode in groups[key]}
    val_eps = {episode for key in val_keys for episode in groups[key]}
    ranges = episode_window_ranges(dataset)
    train_indices = [index for episode in sorted(train_eps) for index in ranges[episode]]
    val_indices = [index for episode in sorted(val_eps) for index in ranges[episode]]
    if not train_indices or not val_indices:
        raise ValueError("Train/validation split produced an empty window set")
    return train_indices, val_indices, train_eps, val_eps, sorted(train_keys), sorted(val_keys)


def audit_temporal_alignment(dataset: TaFDataset, windows_per_episode: int) -> None:
    """Decode representative windows after constructor-level timestamp checks."""
    if dataset.invalid_sequences:
        raise RuntimeError(
            "Dataset contains skipped invalid sequences; refusing to train: "
            + repr(dataset.invalid_sequences)
        )
    if windows_per_episode < 1:
        raise ValueError("alignment_decode_windows_per_episode must be >= 1")
    ranges = episode_window_ranges(dataset)
    decoded = 0
    for episode_id, window_range in enumerate(ranges):
        count = len(window_range)
        offsets = np.linspace(0, count - 1, min(windows_per_episode, count), dtype=int)
        for offset in np.unique(offsets):
            sample = dataset[window_range.start + int(offset)]
            for key in ("tactile", "force", "pressure"):
                if not torch.isfinite(sample[key]).all():
                    raise ValueError(f"Non-finite {key} in episode index {episode_id}")
            metadata = sample["metadata"]
            if not torch.all(torch.diff(metadata["frame_indices"]) == 1):
                raise ValueError(f"Non-contiguous frames in episode index {episode_id}")
            if not torch.all(torch.diff(metadata["timestamps"]) > 0):
                raise ValueError(f"Non-increasing sensor timestamps in episode index {episode_id}")
            if not torch.all(torch.diff(metadata["video_timestamps"]) > 0):
                raise ValueError(f"Non-increasing video timestamps in episode index {episode_id}")
            decoded += 1
    print(f"Temporal alignment audit: PASS ({len(dataset.episodes)} episodes, {decoded} decoded windows)")


def compute_normalization(
    dataset: TaFDataset, train_episode_ids: set[int], chunk_frames: int, eps: float,
) -> dict[str, torch.Tensor]:
    """Compute per-taxel/per-wrench-component population stats from train frames only."""
    if chunk_frames < 1 or eps <= 0:
        raise ValueError("normalization_chunk_frames and normalization_eps must be positive")
    pressure_sum = torch.zeros(12, 12, dtype=torch.float64)
    pressure_square_sum = torch.zeros_like(pressure_sum)
    force_sum = torch.zeros(6, dtype=torch.float64)
    force_square_sum = torch.zeros_like(force_sum)
    count = 0
    for episode_id in sorted(train_episode_ids):
        episode = dataset.episodes[episode_id]
        for start in range(0, episode.length, chunk_frames):
            offsets = np.arange(start, min(start + chunk_frames, episode.length), dtype=np.int64)
            force_array, pressure_array = dataset._sensor_rows(episode, offsets)
            pressure = torch.from_numpy(pressure_array).double()
            force = torch.from_numpy(force_array).double()
            if not torch.isfinite(pressure).all() or not torch.isfinite(force).all():
                raise ValueError(
                    f"Non-finite force data in training episode "
                    f"{episode.sequence_id}/{episode.episode_index}"
                )
            pressure_sum += pressure.sum(0)
            pressure_square_sum += pressure.square().sum(0)
            force_sum += force.sum(0)
            force_square_sum += force.square().sum(0)
            count += len(offsets)
    if count < 2:
        raise ValueError("At least two training frames are required for normalization")
    pressure_mean = pressure_sum / count
    force_mean = force_sum / count
    pressure_std = (
        pressure_square_sum / count - pressure_mean.square()
    ).clamp_min(0).sqrt().clamp_min(eps)
    force_std = (
        force_square_sum / count - force_mean.square()
    ).clamp_min(0).sqrt().clamp_min(eps)
    print(f"Normalization stats: {count} unique training frames (validation excluded)")
    return {
        "pressure_mean": pressure_mean.float(),
        "pressure_std": pressure_std.float(),
        "force_mean": force_mean.float(),
        "force_std": force_std.float(),
    }


class TemporalDiverseBatchSampler(Sampler[list[int]]):
    """Greedily separate nearby same-episode windows without dropping data."""

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
        episode_ids = {id(episode): index for index, episode in enumerate(dataset.episodes)}
        self.items: list[tuple[int, int, int]] = []
        for local_index, global_index in enumerate(subset_indices):
            episode, start = dataset._locate(global_index)
            self.items.append((local_index, episode_ids[id(episode)], start))

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

