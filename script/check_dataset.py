"""Inspect TaF data alignment and smoke-test configurable temporal windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.taf_dataset import TaFDataset


def check_sample(sample: dict, window_size: int) -> None:
    expected = {"tactile": (window_size, 3), "force": (window_size, 6), "pressure": (window_size, 12, 12)}
    for name, prefix in expected.items():
        tensor = sample[name]
        assert tensor.dtype == torch.float32, f"{name} dtype is {tensor.dtype}"
        assert tuple(tensor.shape[:len(prefix)]) == prefix if name == "tactile" else tuple(tensor.shape) == prefix
        assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"
    meta = sample["metadata"]
    frames, timestamps = meta["frame_indices"], meta["timestamps"]
    video_timestamps = meta["video_timestamps"]
    video_frame_indices = meta["video_frame_indices"]
    video_pts = meta["video_pts"]
    assert len(frames) == window_size and len(timestamps) == window_size
    assert torch.all(torch.diff(frames) == 1), f"Non-consecutive frame indices: {frames.tolist()}"
    assert torch.all(torch.diff(timestamps) > 0), f"Non-increasing timestamps: {timestamps.tolist()}"
    assert torch.all(torch.diff(video_timestamps) > 0), "Non-increasing MP4 presentation timestamps"
    assert torch.all(torch.diff(video_frame_indices) > 0), "Non-increasing MP4 frame indices"
    assert torch.all(torch.diff(video_pts) > 0), "Non-increasing decoded PyAV PTS"


def synthetic_window_check() -> None:
    """Small formula check independent of any downloaded dataset."""
    episode_lengths = [2, 5, 9]
    for size, stride, expected in [(1, 1, 16), (3, 1, 10), (5, 2, 4), (10, 1, 0)]:
        actual = sum(max(0, (length - size) // stride + 1) for length in episode_lengths)
        assert actual == expected, (size, stride, actual, expected)
    print("Synthetic window-count and episode-boundary checks: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--test-window-sizes", action="store_true", help="also try T=1,3,5,10,20")
    parser.add_argument("--skip-invalid-sequences", action="store_true")
    parser.add_argument(
        "--timestamp-cache-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse the persistent ffprobe timestamp cache (default: enabled)",
    )
    parser.add_argument(
        "--timestamp-cache-dir",
        type=Path,
        help="cache directory (default: XDG_CACHE_HOME or ~/.cache/taf-vla/video_timestamps)",
    )
    parser.add_argument(
        "--show-initialization-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    synthetic_window_check()
    sizes = [1, 3, 5, 10, 20] if args.test_window_sizes else [args.window_size]
    for size in sizes:
        try:
            ds = TaFDataset(args.root_dir, window_size=size, stride=args.stride,
                            image_size=tuple(args.image_size) if args.image_size else None,
                            max_sequences=args.max_sequences,
                            skip_invalid_sequences=args.skip_invalid_sequences,
                            timestamp_cache_enabled=args.timestamp_cache_enabled,
                            timestamp_cache_dir=args.timestamp_cache_dir,
                            show_initialization_progress=args.show_initialization_progress)
        except ValueError as exc:
            if args.test_window_sizes and "No valid episodes are long enough" in str(exc):
                print(f"T={size}: SKIP ({exc})")
                continue
            raise
        print(f"\nT={size}: sensors={ds.sensor_types}, sequences={ds.sequence_count}, "
              f"episodes={ds.episode_count}, windows={len(ds)}")
        if ds.invalid_sequences:
            print("excluded sequences:", ds.invalid_sequences)
        if ds.reconstructed_episode_metadata:
            print("in-memory reconstructed episode metadata:", ds.reconstructed_episode_metadata)
        sample = ds[0]
        check_sample(sample, size)
        print("sample:", {key: (tuple(value.shape), str(value.dtype)) for key, value in sample.items() if key != "metadata"})
        print("alignment:", sample["metadata"]["frame_indices"].tolist(),
              sample["metadata"]["timestamps"].tolist(), "video",
              sample["metadata"]["video_timestamps"].tolist(), "PTS",
              sample["metadata"]["video_pts"].tolist(), "episode",
              sample["metadata"]["episode_index"])
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        batch = next(iter(loader))
        assert tuple(batch["tactile"].shape[:3]) == (args.batch_size, size, 3)
        assert tuple(batch["force"].shape) == (args.batch_size, size, 6)
        assert tuple(batch["pressure"].shape) == (args.batch_size, size, 12, 12)
        print("batch shapes:", tuple(batch["tactile"].shape), tuple(batch["force"].shape), tuple(batch["pressure"].shape))
        print(f"PyAV DataLoader workers={args.num_workers}: PASS")
        # Check boundaries for first, last, and episode-transition windows.
        boundary_indices = {0, len(ds) - 1}
        for end in ds._ends[:-1]:
            boundary_indices.update((end - 1, end))
        for idx in boundary_indices:
            if idx < 0 or idx >= len(ds):
                continue
            item = ds[idx]
            check_sample(item, size)
            ep, _ = ds._locate(idx)
            assert item["metadata"]["episode_index"] == ep.episode_index
        print("continuity, finite-value, and episode-boundary checks: PASS")


if __name__ == "__main__":
    main()
