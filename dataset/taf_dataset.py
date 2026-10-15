"""PyTorch Dataset for temporally aligned LeRobot v3 TaF data."""

from __future__ import annotations

import bisect
import json
import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class _Episode:
    sensor_type: str
    sequence_id: str
    episode_index: int
    parquet: Path
    video: Path
    row_start: int
    length: int
    frames: np.ndarray
    timestamps: np.ndarray
    video_timestamps: np.ndarray
    fps: float


def _visible_files(directory: Path, suffix: str) -> list[Path]:
    files: list[Path] = []
    if directory.is_dir():
        for current, dirs, names in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            files.extend(Path(current) / n for n in names if not n.startswith(".") and n.endswith(suffix))
    return sorted(files)


def _column_matrix(column: Any, name: str) -> np.ndarray:
    array = column.combine_chunks()
    values = getattr(array, "values", None)
    if values is not None:
        try:
            return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(array), -1)
        except (TypeError, ValueError):
            pass
    try:
        return np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in array.to_pylist()])
    except ValueError as exc:
        raise ValueError(f"Ragged values in Parquet column {name}") from exc


def _format_path(template: str, *, video_key: str | None, chunk: int, file: int) -> str:
    return template.format(video_key=video_key, chunk_index=chunk, file_index=file)


def _probe_video_timestamps(path: Path) -> np.ndarray:
    """Return presentation timestamps for every decoded video frame via ffprobe."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to map Parquet timestamps to MP4 frames") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc
    values = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.split(",")[0]))
        except ValueError:
            continue
    timestamps = np.asarray(values, dtype=np.float64)
    if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"No valid increasing frame timestamps found in {path}")
    return timestamps


class TaFDataset(Dataset):
    """Lazy temporal windows resolved from official LeRobot v3 episode metadata."""

    SENSOR_COLUMNS = ("observation.force_torque", "observation.pressure_matrix")

    def __init__(
        self,
        root_dir: str | Path,
        window_size: int = 5,
        stride: int = 1,
        image_size: tuple[int, int] | int | None = None,
        sensor_types: list[str] | tuple[str, ...] | str | None = None,
        max_sequences: int | None = None,
        skip_invalid_sequences: bool = False,
    ) -> None:
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must both be >= 1")
        if max_sequences is not None and max_sequences < 1:
            raise ValueError("max_sequences must be >= 1")
        if isinstance(image_size, int) and image_size < 1:
            raise ValueError("image_size must be positive")
        if isinstance(image_size, tuple) and (len(image_size) != 2 or min(image_size) < 1):
            raise ValueError("image_size must be a positive int or (height, width)")
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.window_size, self.stride, self.image_size = int(window_size), int(stride), image_size
        self.skip_invalid_sequences = skip_invalid_sequences
        if isinstance(sensor_types, str):
            sensor_types = [sensor_types]
        allowed = set(sensor_types) if sensor_types else None
        taf_root = self.root_dir if self.root_dir.name == "taf_dataset" else self.root_dir / "taf_dataset"
        if not taf_root.is_dir():
            raise FileNotFoundError(f"Expected TaF data under {taf_root}")
        candidates = []
        for sensor_dir in sorted(taf_root.iterdir()):
            if sensor_dir.is_dir() and not sensor_dir.name.startswith("."):
                candidates.extend(p for p in sorted(sensor_dir.iterdir())
                                  if p.is_dir() and not p.name.startswith(".") and re.search(r"_obj\d+$", p.name))
        if max_sequences is not None:
            candidates = candidates[:max_sequences]
        if not candidates:
            raise FileNotFoundError(f"No Sequence directories matching '*_obj<ID>' under {taf_root}")

        self.episodes: list[_Episode] = []
        self.invalid_sequences: list[tuple[str, str]] = []
        self._video_timestamp_cache: dict[Path, np.ndarray] = {}
        for directory in candidates:
            match = re.match(r"(.+)_obj\d+$", directory.name)
            sensor = match.group(1)
            if allowed and sensor not in allowed:
                continue
            try:
                self.episodes.extend(self._load_sequence(directory, sensor))
            except Exception as exc:
                message = f"{directory.name}: {exc}"
                if not skip_invalid_sequences:
                    raise ValueError(f"Invalid TaF Sequence {message}") from exc
                self.invalid_sequences.append((directory.name, str(exc)))
                warnings.warn(f"Skipping invalid TaF Sequence {message}", RuntimeWarning)

        if self.invalid_sequences:
            warnings.warn(
                f"Skipped {len(self.invalid_sequences)} invalid sequence(s): "
                + "; ".join(f"{name} ({reason})" for name, reason in self.invalid_sequences),
                RuntimeWarning,
            )
        if not self.episodes:
            raise ValueError("No valid episodes remain after sequence validation")
        self.episodes = [ep for ep in self.episodes if ep.length >= self.window_size]
        if not self.episodes:
            raise ValueError(f"No valid episodes are long enough for window_size={self.window_size}")
        self._ends: list[int] = []
        total = 0
        for ep in self.episodes:
            total += (ep.length - self.window_size) // self.stride + 1
            self._ends.append(total)
        self._sensor_cache: tuple[Path, dict[str, np.ndarray]] | None = None
        self._video = None
        self._video_path: Path | None = None

    def _load_sequence(self, directory: Path, sensor: str) -> list[_Episode]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("TaFDataset requires pyarrow") from exc
        info_path = directory / "meta" / "info.json"
        episode_files = _visible_files(directory / "meta" / "episodes", ".parquet")
        if not info_path.is_file() or not episode_files:
            raise FileNotFoundError("requires meta/info.json and meta/episodes/**/*.parquet")
        info = json.loads(info_path.read_text())
        data_template = info.get("data_path")
        video_template = info.get("video_path")
        if not data_template or not video_template:
            raise ValueError("meta/info.json lacks data_path or video_path templates")
        import pyarrow as pa
        episode_table = pa.concat_tables([pq.read_table(path) for path in episode_files])
        rows = sorted(episode_table.to_pylist(), key=lambda row: int(row["episode_index"]))
        output = []
        data_index_cache: dict[Path, Any] = {}
        for meta in rows:
            episode_index = int(meta["episode_index"])
            length = int(meta["length"])
            data_path = directory / _format_path(data_template,
                video_key=None, chunk=int(meta["data/chunk_index"]), file=int(meta["data/file_index"]))
            video_key = "observation.image"
            video_path = directory / _format_path(video_template, video_key=video_key,
                chunk=int(meta[f"videos/{video_key}/chunk_index"]),
                file=int(meta[f"videos/{video_key}/file_index"]))
            if not data_path.is_file() or not video_path.is_file():
                raise FileNotFoundError(f"episode {episode_index} references missing file: {data_path} or {video_path}")
            required = {"index", "episode_index", "frame_index", "timestamp", *TaFDataset.SENSOR_COLUMNS}
            schema = set(pq.read_schema(data_path).names)
            missing = required - schema
            if missing:
                raise ValueError(f"{data_path} lacks columns {sorted(missing)}")
            start, stop = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
            if stop - start != length:
                raise ValueError(f"episode {episode_index}: dataset index range length {stop-start} != metadata length {length}")
            if data_path not in data_index_cache:
                data_index_cache[data_path] = pq.read_table(
                    data_path, columns=["index", "episode_index", "frame_index", "timestamp"]
                )
            table = data_index_cache[data_path]
            indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
            positions = np.flatnonzero((indices >= start) & (indices < stop))
            if len(positions) != length or not np.array_equal(indices[positions], np.arange(start, stop)):
                raise ValueError(f"episode {episode_index}: data rows do not match dataset_from_index/to_index")
            ep_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)[positions]
            if not np.all(ep_ids == episode_index):
                raise ValueError(f"episode {episode_index}: referenced data range contains another episode")
            frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)[positions]
            timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)[positions]
            if np.any(np.diff(frames) != 1) or np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"episode {episode_index}: frame_index/timestamp is not strictly contiguous")
            video_from = float(meta[f"videos/{video_key}/from_timestamp"])
            video_to = float(meta[f"videos/{video_key}/to_timestamp"])
            target_times = video_from + timestamps
            if np.any(target_times < video_from - 1e-6) or np.any(target_times >= video_to + 1e-6):
                raise ValueError(f"episode {episode_index}: Parquet timestamps fall outside video timestamp range [{video_from}, {video_to})")
            fps = float(info.get("fps", 30))
            video_times = self._video_times(video_path)
            nearest = np.searchsorted(video_times, target_times).clip(0, len(video_times) - 1)
            left = np.maximum(nearest - 1, 0)
            nearest = np.where(np.abs(video_times[left] - target_times) <=
                               np.abs(video_times[nearest] - target_times), left, nearest)
            if np.any(np.abs(video_times[nearest] - target_times) > 0.5 / fps + 1e-3):
                raise ValueError(f"episode {episode_index}: Parquet timestamps cannot be matched to MP4 frames")
            if len(np.unique(nearest)) != length:
                raise ValueError(f"episode {episode_index}: multiple Parquet rows map to the same MP4 frame")
            output.append(_Episode(sensor, directory.name, episode_index, data_path, video_path,
                                   int(positions[0]), length, frames, timestamps, target_times,
                                   fps))
        return output

    def _video_times(self, path: Path) -> np.ndarray:
        if path not in self._video_timestamp_cache:
            self._video_timestamp_cache[path] = _probe_video_timestamps(path)
        return self._video_timestamp_cache[path]

    def __len__(self) -> int:
        return self._ends[-1]

    @property
    def sequence_count(self) -> int:
        return len({ep.sequence_id for ep in self.episodes})

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def sensor_types(self) -> list[str]:
        return sorted({ep.sensor_type for ep in self.episodes})

    def _locate(self, index: int) -> tuple[_Episode, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep_idx = bisect.bisect_right(self._ends, index)
        previous = self._ends[ep_idx - 1] if ep_idx else 0
        return self.episodes[ep_idx], (index - previous) * self.stride

    def _sensor_rows(self, ep: _Episode, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import pyarrow.parquet as pq
        if self._sensor_cache is None or self._sensor_cache[0] != ep.parquet:
            table = pq.read_table(ep.parquet, columns=["index", "observation.force_torque", "observation.pressure_matrix"])
            self._sensor_cache = (ep.parquet, {
                "index": np.asarray(table["index"].to_numpy(), dtype=np.int64),
                "force": _column_matrix(table["observation.force_torque"], "observation.force_torque"),
                "pressure": _column_matrix(table["observation.pressure_matrix"], "observation.pressure_matrix"),
            })
        cache = self._sensor_cache[1]
        # Episode row offset came from metadata's global index range, not the video ordinal.
        expected_indices = cache["index"][ep.row_start] + offsets
        positions = np.searchsorted(cache["index"], expected_indices)
        if np.any(positions >= len(cache["index"])) or not np.array_equal(cache["index"][positions], expected_indices):
            raise ValueError(f"Episode {ep.episode_index} row mapping does not match dataset index")
        force = cache["force"][positions]
        pressure = cache["pressure"][positions]
        if force.shape[1] != 6 or pressure.shape[1] != 144:
            raise ValueError(f"Unexpected sensor shapes: force={force.shape}, pressure={pressure.shape}")
        return force.astype(np.float32, copy=False), pressure.reshape(-1, 12, 12).astype(np.float32, copy=False)

    def _images(self, ep: _Episode, offsets: np.ndarray) -> np.ndarray:
        import cv2
        video_times = self._video_times(ep.video)
        target = ep.video_timestamps[offsets]
        positions = np.searchsorted(video_times, target)
        left = np.clip(positions - 1, 0, len(video_times) - 1)
        right = np.clip(positions, 0, len(video_times) - 1)
        nearest = np.where(np.abs(video_times[left] - target) <= np.abs(video_times[right] - target), left, right)
        deltas = np.abs(video_times[nearest] - target)
        if np.any(deltas > 0.5 / ep.fps + 1e-3):
            i = int(np.argmax(deltas))
            raise ValueError(f"episode {ep.episode_index}: no MP4 frame within tolerance of Parquet timestamp {target[i]:.6f}s")
        if len(np.unique(nearest)) != len(nearest):
            raise ValueError(f"episode {ep.episode_index}: multiple Parquet rows map to the same MP4 frame")
        if self._video_path != ep.video:
            if self._video is not None:
                self._video.release()
            self._video = cv2.VideoCapture(str(ep.video))
            if not self._video.isOpened():
                raise ValueError(f"Could not open video {ep.video}")
            self._video_path = ep.video
        images = []
        for frame_number in nearest:
            self._video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, bgr = self._video.read()
            if not ok:
                raise ValueError(f"Could not decode frame {frame_number} from {ep.video}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.image_size is not None:
                size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
                rgb = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            images.append(rgb)
        return np.stack(images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ep, start = self._locate(index)
        offsets = np.arange(start, start + self.window_size, dtype=np.int64)
        if offsets[-1] >= ep.length:
            raise RuntimeError("Window crossed an episode boundary")
        force, pressure = self._sensor_rows(ep, offsets)
        images = self._images(ep, offsets)
        return {
            "tactile": torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float().div_(255),
            "force": torch.from_numpy(force),
            "pressure": torch.from_numpy(pressure),
            "metadata": {
                "sensor_type": ep.sensor_type, "sequence_id": ep.sequence_id,
                "episode_index": ep.episode_index,
                "frame_indices": torch.from_numpy(ep.frames[offsets].copy()),
                "timestamps": torch.from_numpy(ep.timestamps[offsets].copy()),
                "video_timestamps": torch.from_numpy(ep.video_timestamps[offsets].copy()),
            },
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video"] = None
        state["_video_path"] = None
        state["_sensor_cache"] = None
        state["_video_timestamp_cache"] = {}
        return state
