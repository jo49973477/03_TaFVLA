"""Create a tiny two-sequence TaF-shaped dataset for trainer smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def create_sequence(root: Path, name: str, seed: int, frames: int = 8) -> None:
    sequence = root / "taf_dataset" / "synthetic" / name
    data_path = sequence / "data" / "chunk-0" / "file-0.parquet"
    video_path = sequence / "videos" / "observation.image" / "chunk-0" / "file-0.mp4"
    episode_path = sequence / "meta" / "episodes" / "chunk-0" / "file-0.parquet"
    for path in (data_path, video_path, episode_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    timestamps = np.arange(frames, dtype=np.float64) / 30.0
    force = rng.normal(size=(frames, 6)).astype(np.float32)
    pressure = rng.normal(size=(frames, 144)).astype(np.float32)
    table = pa.table({
        "index": pa.array(np.arange(frames), type=pa.int64()),
        "episode_index": pa.array(np.zeros(frames, dtype=np.int64)),
        "frame_index": pa.array(np.arange(frames), type=pa.int64()),
        "timestamp": pa.array(timestamps),
        "observation.force_torque": pa.array(force.tolist(), type=pa.list_(pa.float32(), 6)),
        "observation.pressure_matrix": pa.array(
            pressure.tolist(), type=pa.list_(pa.float32(), 144),
        ),
    })
    pq.write_table(table, data_path)

    episode = pa.table({
        "episode_index": [0],
        "length": [frames],
        "dataset_from_index": [0],
        "dataset_to_index": [frames],
        "data/chunk_index": [0],
        "data/file_index": [0],
        "videos/observation.image/chunk_index": [0],
        "videos/observation.image/file_index": [0],
        "videos/observation.image/from_timestamp": [0.0],
        "videos/observation.image/to_timestamp": [frames / 30.0],
    })
    pq.write_table(episode, episode_path)

    info = {
        "fps": 30,
        "data_path": "data/chunk-{chunk_index}/file-{file_index}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index}/file-{file_index}.mp4",
    }
    (sequence / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic video {video_path}")
    try:
        for index in range(frames):
            image = np.full((32, 32, 3), index * 20, dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    create_sequence(args.output, "sequence_a_obj1", seed=1)
    create_sequence(args.output, "sequence_b_obj2", seed=2)
    print(f"Synthetic TaF dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
