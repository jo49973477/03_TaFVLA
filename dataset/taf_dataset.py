"""PyTorch Dataset for temporally aligned LeRobot v3 TaF data."""

from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


_TIMESTAMP_CACHE_FORMAT_VERSION = 1
_TIMESTAMP_EXTRACTION_STRATEGY_VERSION = "ffprobe-best-effort-timestamp-time-v1"


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
    video_frame_indices: np.ndarray
    fps: float


@dataclass
class _SequencePlan:
    directory: Path
    sensor: str
    info: dict[str, Any]
    rows: list[dict[str, Any]]


def _validate_video_timestamps(timestamps: np.ndarray, path: Path) -> np.ndarray:
    """캐시와 ffprobe 결과가 동일한 1차원 float64 PTS 배열인지 검증한다."""
    if not isinstance(timestamps, np.ndarray):
        raise ValueError(f"Timestamp cache for {path} is not a NumPy array")
    if timestamps.dtype != np.float64 or timestamps.ndim != 1:
        raise ValueError(
            f"Timestamp cache for {path} must be a 1D float64 array, "
            f"got dtype={timestamps.dtype}, shape={timestamps.shape}"
        )
    if timestamps.size == 0:
        raise ValueError(f"Timestamp array is empty for {path}")
    if not np.isfinite(timestamps).all():
        raise ValueError(f"Timestamp array contains NaN/Inf for {path}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Frame timestamps are not strictly increasing for {path}")
    return timestamps


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
    return _validate_video_timestamps(timestamps, path)


def _default_timestamp_cache_dir() -> Path:
    """XDG 규칙을 따르는 원본 데이터셋 외부의 기본 캐시 경로를 반환한다."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "taf-vla" / "video_timestamps"


class _PersistentTimestampCache:
    """영상별 ffprobe 결과를 검증 가능한 단일 NPZ 파일로 원자 저장한다.

    각 영상 경로의 SHA-256이 파일명이 된다. 동일 파일명의 lock을 잡고
    캐시를 다시 확인한 뒤에만 ffprobe를 실행하므로, 여러 학습 프로세스가
    동시에 시작되어도 완성되지 않은 캐시를 읽거나 같은 캐시를 덮어쓰는
    경쟁을 피한다.
    """

    def __init__(self, enabled: bool, cache_dir: str | Path | None) -> None:
        self.enabled = bool(enabled)
        selected = _default_timestamp_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
        self.cache_dir = selected.resolve()
        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                warnings.warn(
                    f"Timestamp cache directory {self.cache_dir} is unavailable; "
                    f"continuing without persistent cache: {exc}",
                    RuntimeWarning,
                )
                self.enabled = False

    @staticmethod
    def _identity(path: Path) -> dict[str, Any]:
        canonical = path.resolve(strict=True)
        stat = canonical.stat()
        return {
            "canonical_path": str(canonical),
            "file_size": int(stat.st_size),
            "st_mtime_ns": int(stat.st_mtime_ns),
            "cache_format_version": _TIMESTAMP_CACHE_FORMAT_VERSION,
            "timestamp_extraction_strategy_version": _TIMESTAMP_EXTRACTION_STRATEGY_VERSION,
        }

    @staticmethod
    def _key(identity: dict[str, Any]) -> str:
        return hashlib.sha256(identity["canonical_path"].encode("utf-8")).hexdigest()

    def _paths(self, identity: dict[str, Any]) -> tuple[Path, Path]:
        key = self._key(identity)
        return self.cache_dir / f"{key}.npz", self.cache_dir / f"{key}.lock"

    def _load(self, cache_path: Path, identity: dict[str, Any]) -> np.ndarray | None:
        if not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as stored:
                if set(stored.files) != {"timestamps", "metadata"}:
                    raise ValueError("unexpected NPZ members")
                metadata_array = stored["metadata"]
                if metadata_array.shape != () or metadata_array.dtype.kind not in {"U", "S"}:
                    raise ValueError("metadata must be a scalar string")
                metadata = json.loads(str(metadata_array.item()))
                timestamps = np.array(stored["timestamps"], copy=True)
            expected_checksum = metadata.pop("timestamps_sha256", None)
            if metadata != identity:
                return None
            timestamps = _validate_video_timestamps(timestamps, Path(identity["canonical_path"]))
            actual_checksum = hashlib.sha256(timestamps.tobytes(order="C")).hexdigest()
            if expected_checksum != actual_checksum:
                raise ValueError("timestamp checksum mismatch")
            return timestamps
        except Exception as exc:
            warnings.warn(
                f"Ignoring invalid timestamp cache {cache_path}; it will be regenerated: {exc}",
                RuntimeWarning,
            )
            return None

    def _store(self, cache_path: Path, identity: dict[str, Any], timestamps: np.ndarray) -> None:
        metadata = dict(identity)
        metadata["timestamps_sha256"] = hashlib.sha256(
            timestamps.tobytes(order="C")
        ).hexdigest()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{cache_path.stem}.",
                suffix=".tmp",
                dir=self.cache_dir,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez(
                    temporary,
                    timestamps=timestamps,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, cache_path)
            temporary_path = None
            try:
                directory_fd = os.open(self.cache_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # 파일의 atomic replace는 이미 끝났다. 디렉터리 fsync를 지원하지
                # 않는 파일시스템이라는 이유만으로 정상 캐시를 폐기하지 않는다.
                pass
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load_if_valid(self, path: Path) -> np.ndarray | None:
        """ffprobe를 실행하지 않고 현재 파일과 일치하는 캐시만 반환한다."""
        if not self.enabled:
            return None
        try:
            identity = self._identity(path)
        except OSError:
            return None
        cache_path, _ = self._paths(identity)
        return self._load(cache_path, identity)

    def get(self, path: Path) -> tuple[np.ndarray, bool]:
        """Timestamp와 persistent-cache hit 여부를 반환한다."""
        identity = self._identity(path)
        canonical = Path(identity["canonical_path"])
        if not self.enabled:
            return _probe_video_timestamps(canonical), False

        cache_path, lock_path = self._paths(identity)
        cached = self._load(cache_path, identity)
        if cached is not None:
            return cached, True

        try:
            with lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                # Lock 대기 중 다른 프로세스가 캐시를 완성했을 수 있다.
                identity = self._identity(canonical)
                cache_path, _ = self._paths(identity)
                cached = self._load(cache_path, identity)
                if cached is not None:
                    return cached, True

                before = identity
                timestamps = _probe_video_timestamps(canonical)
                after = self._identity(canonical)
                if before != after:
                    raise RuntimeError(f"Video changed while ffprobe was reading it: {canonical}")
                try:
                    self._store(cache_path, identity, timestamps)
                except OSError as exc:
                    warnings.warn(
                        f"Could not persist timestamp cache for {canonical}: {exc}",
                        RuntimeWarning,
                    )
                return timestamps, False
        except OSError as exc:
            warnings.warn(
                f"Could not lock timestamp cache for {canonical}; probing without disk cache: {exc}",
                RuntimeWarning,
            )
            return _probe_video_timestamps(canonical), False


class _InitializationProgress:
    """Dataset 초기화 전용 고정 위치 tqdm UI와 non-TTY fallback을 관리한다."""

    def __init__(
        self,
        sequence_total: int,
        video_total: int,
        enabled: bool,
        *,
        stream: TextIO | None = None,
        force_tty: bool | None = None,
        initial_cache_hits: int = 0,
        cache_scan_complete: bool = False,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        detected_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.dynamic = bool(enabled) and (detected_tty if force_tty is None else force_tty)
        self.enabled = bool(enabled)
        self.sequence_total = sequence_total
        self.video_total = video_total
        self.sequences = 0
        self._misses_precounted = bool(cache_scan_complete)
        self.videos = int(initial_cache_hits)
        self.hits = int(initial_cache_hits)
        self.misses = video_total - self.hits if self._misses_precounted else 0
        self.failed = 0
        self._plain_interval = max(1, video_total // 10) if video_total else 1
        self._sequence_bar = None
        self._video_bar = None
        self._status_bar = None
        if not self.enabled:
            return
        self.stream.write("TaF Dataset Initialization\n")
        if self._misses_precounted:
            self.stream.write(f"Found {self.hits} valid cached video(s).\n")
            self.stream.write(f"Remaining {self.misses} video(s) require ffprobe.\n")
        self.stream.flush()
        if self.dynamic:
            self._sequence_bar = tqdm(
                total=sequence_total,
                desc="Loading sequences",
                unit="seq",
                position=0,
                leave=True,
                dynamic_ncols=True,
                file=self.stream,
            )
            self._video_bar = tqdm(
                total=video_total,
                initial=self.videos,
                desc="Videos",
                unit="video",
                position=1,
                leave=True,
                dynamic_ncols=True,
                file=self.stream,
            )
            self._status_bar = tqdm(
                total=0,
                bar_format="{desc}",
                position=2,
                leave=True,
                file=self.stream,
            )
            self._refresh_status()
        else:
            self.stream.write(
                f"Loading {sequence_total} sequence(s); "
                f"{video_total} metadata-referenced video(s).\n"
            )
            self.stream.flush()

    def start_sequence(self, name: str) -> None:
        if self._sequence_bar is not None:
            self._sequence_bar.set_description_str(f"Loading sequences ({name})", refresh=True)

    def finish_sequence(self) -> None:
        self.sequences += 1
        if self._sequence_bar is not None:
            self._sequence_bar.update(1)

    def start_video(self, path: Path) -> None:
        if self._video_bar is not None:
            self._video_bar.set_description_str(f"Videos ({path.name})", refresh=True)

    def finish_video(self, *, cache_hit: bool) -> None:
        self.videos += 1
        if self._misses_precounted:
            if cache_hit:
                # 사전 검사 이후 다른 프로세스가 캐시를 생성한 경우다.
                self.hits += 1
                self.misses -= 1
        else:
            if cache_hit:
                self.hits += 1
            else:
                self.misses += 1
        if self._video_bar is not None:
            self._video_bar.update(1)
            self._refresh_status()
        elif self.enabled and (
            self.videos == self.video_total or self.videos % self._plain_interval == 0
        ):
            self.stream.write(
                f"Timestamp videos {self.videos}/{self.video_total} | "
                f"cache hits={self.hits}, misses={self.misses}, failed={self.failed}\n"
            )
            self.stream.flush()

    def fail_video(self) -> None:
        self.videos += 1
        if not self._misses_precounted:
            self.misses += 1
        self.failed += 1
        if self._video_bar is not None:
            self._video_bar.update(1)
            self._refresh_status()

    def _refresh_status(self) -> None:
        if self._status_bar is not None:
            self._status_bar.set_description_str(
                f"Cache hits: {self.hits} | misses: {self.misses} | failed videos: {self.failed}",
                refresh=True,
            )

    def close(self) -> None:
        for bar in (self._status_bar, self._video_bar, self._sequence_bar):
            if bar is not None:
                bar.close()
        if self.enabled and not self.dynamic:
            self.stream.write(
                f"Dataset initialization finished: sequences={self.sequences}/{self.sequence_total}, "
                f"videos={self.videos}/{self.video_total}, cache hits={self.hits}, "
                f"misses={self.misses}, failed={self.failed}\n"
            )
            self.stream.flush()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "failed_videos": self.failed,
            "processed_videos": self.videos,
            "total_videos": self.video_total,
        }


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
        timestamp_cache_enabled: bool = True,
        timestamp_cache_dir: str | Path | None = None,
        show_initialization_progress: bool = True,
    ) -> None:
        # FFmpeg/PyAV may initialize native decoder state when the main process
        # audits a sample. A later POSIX fork can inherit locked native state.
        # Select spawn before any DataLoader context exists; __getstate__ below
        # removes all process-local handles from the worker copy.
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
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
        self.timestamp_cache_enabled = bool(timestamp_cache_enabled)
        self.timestamp_cache_dir = timestamp_cache_dir
        self.show_initialization_progress = bool(show_initialization_progress)
        self._timestamp_store = _PersistentTimestampCache(
            self.timestamp_cache_enabled,
            timestamp_cache_dir,
        )
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
        self.reconstructed_episode_metadata: list[str] = []
        self._video_timestamp_cache: dict[Path, np.ndarray] = {}
        plans: list[_SequencePlan] = []
        for directory in candidates:
            match = re.match(r"(.+)_obj\d+$", directory.name)
            sensor = match.group(1)
            if allowed and sensor not in allowed:
                continue
            try:
                plans.append(self._prepare_sequence(directory, sensor))
            except Exception as exc:
                self._handle_invalid_sequence(directory, exc)

        referenced_videos = {
            self._metadata_video_path(plan.directory, plan.info, meta)
            for plan in plans
            for meta in plan.rows
        }
        # 디스크 캐시만 빠르게 검사한다. 여기서는 ffprobe를 절대 실행하지
        # 않으며, 정상 timestamp 배열은 뒤의 sequence validation에서 재사용한다.
        for video_path in sorted(referenced_videos):
            cached = self._timestamp_store.load_if_valid(video_path)
            if cached is not None:
                self._video_timestamp_cache[video_path] = cached
        progress = _InitializationProgress(
            sequence_total=len(plans),
            video_total=len(referenced_videos),
            enabled=self.show_initialization_progress,
            initial_cache_hits=len(self._video_timestamp_cache),
            cache_scan_complete=True,
        )
        self._initialization_progress = progress
        try:
            for plan in plans:
                progress.start_sequence(plan.directory.name)
                try:
                    self.episodes.extend(self._load_sequence(plan))
                except Exception as exc:
                    self._handle_invalid_sequence(plan.directory, exc)
                finally:
                    progress.finish_sequence()
        finally:
            progress.close()
            self.initialization_stats = progress.stats
            self._initialization_progress = None

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
        self._video_stream = None
        self._video_path: Path | None = None
        self._video_pid: int | None = None

    def _handle_invalid_sequence(self, directory: Path, exc: Exception) -> None:
        """기존 skip_invalid_sequences 정책을 metadata 준비와 본 로딩에 공통 적용한다."""
        message = f"{directory.name}: {exc}"
        if not self.skip_invalid_sequences:
            raise ValueError(f"Invalid TaF Sequence {message}") from exc
        self.invalid_sequences.append((directory.name, str(exc)))
        warnings.warn(f"Skipping invalid TaF Sequence {message}", RuntimeWarning)

    def _prepare_sequence(self, directory: Path, sensor: str) -> _SequencePlan:
        """ffprobe 없이 LeRobot metadata를 읽어 대상 영상 목록을 확정한다."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("TaFDataset requires pyarrow") from exc
        info_path = directory / "meta" / "info.json"
        episode_files = _visible_files(directory / "meta" / "episodes", ".parquet")
        if not info_path.is_file():
            raise FileNotFoundError("requires meta/info.json")
        info = json.loads(info_path.read_text())
        data_template = info.get("data_path")
        video_template = info.get("video_path")
        if not data_template or not video_template:
            raise ValueError("meta/info.json lacks data_path or video_path templates")
        if episode_files:
            import pyarrow as pa
            episode_table = pa.concat_tables([pq.read_table(path) for path in episode_files])
            rows = sorted(episode_table.to_pylist(), key=lambda row: int(row["episode_index"]))
        else:
            rows = self._reconstruct_episode_rows(directory, info)
            self.reconstructed_episode_metadata.append(directory.name)
            warnings.warn(
                f"{directory.name}: meta/episodes is missing; reconstructed episode "
                "boundaries in memory from validated Parquet episode_index/frame_index "
                "columns. The source dataset was not modified.",
                RuntimeWarning,
            )
        # 총 영상 수는 실제 Dataset metadata가 참조하는 파일만 집계한다.
        for meta in rows:
            self._metadata_video_path(directory, info, meta)
        return _SequencePlan(directory=directory, sensor=sensor, info=info, rows=rows)

    @staticmethod
    def _chunk_file_indices(path: Path) -> tuple[int, int]:
        """LeRobot의 chunk-N/file-N 경로에서 두 정수 index를 읽는다."""
        chunk_match = re.fullmatch(r"chunk-(\d+)", path.parent.name)
        file_match = re.fullmatch(r"file-(\d+)", path.stem)
        if chunk_match is None or file_match is None:
            raise ValueError(f"Cannot infer LeRobot chunk/file indices from {path}")
        return int(chunk_match.group(1)), int(file_match.group(1))

    def _reconstruct_episode_rows(
        self,
        directory: Path,
        info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """누락된 meta/episodes를 단일-file LeRobot Parquet에서 엄격히 복원한다.

        이 fallback은 원본 파일을 만들거나 수정하지 않는다. 센서 Parquet의
        global index, episode_index, local frame_index와 timestamp가 모두
        일관적인 경우에만 in-memory metadata를 만든다. 생성된 video offset은
        정상 TaF metadata와 동일하게 누적 frame count / fps로 계산되며, 뒤의
        기존 ffprobe half-frame tolerance 검증을 그대로 통과해야 한다.
        """
        import pyarrow.parquet as pq

        data_files = _visible_files(directory / "data", ".parquet")
        video_files = _visible_files(directory / "videos" / "observation.image", ".mp4")
        if len(data_files) != 1 or len(video_files) != 1:
            raise FileNotFoundError(
                "meta/episodes is missing and safe reconstruction requires exactly "
                f"one data Parquet and one observation.image MP4; found "
                f"data={len(data_files)}, videos={len(video_files)}"
            )

        data_path = data_files[0]
        video_path = video_files[0]
        data_chunk, data_file = self._chunk_file_indices(data_path)
        video_chunk, video_file = self._chunk_file_indices(video_path)
        expected_data = directory / _format_path(
            info["data_path"], video_key=None, chunk=data_chunk, file=data_file,
        )
        expected_video = directory / _format_path(
            info["video_path"],
            video_key="observation.image",
            chunk=video_chunk,
            file=video_file,
        )
        if expected_data.resolve() != data_path.resolve() or expected_video.resolve() != video_path.resolve():
            raise ValueError("Discovered files do not match info.json path templates")

        table = pq.read_table(
            data_path,
            columns=["index", "episode_index", "frame_index", "timestamp"],
        )
        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        if len(indices) == 0 or not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError("Cannot reconstruct episodes: global index must be contiguous from zero")
        if not np.isfinite(timestamps).all():
            raise ValueError("Cannot reconstruct episodes: timestamps contain NaN/Inf")

        unique_episodes = np.unique(episode_ids)
        expected_episode_ids = np.arange(len(unique_episodes), dtype=np.int64)
        if not np.array_equal(unique_episodes, expected_episode_ids):
            raise ValueError("Cannot reconstruct episodes: episode_index must be contiguous from zero")
        if int(info.get("total_frames", -1)) != len(indices):
            raise ValueError("Cannot reconstruct episodes: info.total_frames does not match Parquet")
        if int(info.get("total_episodes", -1)) != len(unique_episodes):
            raise ValueError("Cannot reconstruct episodes: info.total_episodes does not match Parquet")

        fps = float(info.get("fps", 0))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot reconstruct episodes: info.fps must be positive")
        rows: list[dict[str, Any]] = []
        cumulative_frames = 0
        for episode_index in unique_episodes:
            positions = np.flatnonzero(episode_ids == episode_index)
            start = int(positions[0])
            stop = int(positions[-1]) + 1
            if not np.array_equal(positions, np.arange(start, stop)):
                raise ValueError(f"episode {episode_index}: rows are not contiguous")
            length = stop - start
            expected_frames = np.arange(length, dtype=np.int64)
            if not np.array_equal(frame_indices[positions], expected_frames):
                raise ValueError(f"episode {episode_index}: frame_index must restart at zero")
            local_timestamps = timestamps[positions]
            if abs(float(local_timestamps[0])) > 1e-6:
                raise ValueError(f"episode {episode_index}: timestamp must restart at zero")
            if length > 1 and np.any(np.diff(local_timestamps) <= 0):
                raise ValueError(f"episode {episode_index}: timestamps are not increasing")
            expected_times = expected_frames.astype(np.float64) / fps
            if np.any(np.abs(local_timestamps - expected_times) > 0.5 / fps + 1e-3):
                raise ValueError(f"episode {episode_index}: timestamps are inconsistent with fps")

            video_from = cumulative_frames / fps
            cumulative_frames += length
            rows.append({
                "episode_index": int(episode_index),
                "length": length,
                "data/chunk_index": data_chunk,
                "data/file_index": data_file,
                "dataset_from_index": start,
                "dataset_to_index": stop,
                "videos/observation.image/chunk_index": video_chunk,
                "videos/observation.image/file_index": video_file,
                "videos/observation.image/from_timestamp": video_from,
                "videos/observation.image/to_timestamp": cumulative_frames / fps,
            })
        return rows

    @staticmethod
    def _metadata_video_path(
        directory: Path,
        info: dict[str, Any],
        meta: dict[str, Any],
    ) -> Path:
        video_key = "observation.image"
        path = directory / _format_path(
            info["video_path"],
            video_key=video_key,
            chunk=int(meta[f"videos/{video_key}/chunk_index"]),
            file=int(meta[f"videos/{video_key}/file_index"]),
        )
        return path.resolve()

    def _load_sequence(self, plan: _SequencePlan) -> list[_Episode]:
        import pyarrow.parquet as pq

        directory = plan.directory
        sensor = plan.sensor
        info = plan.info
        rows = plan.rows
        data_template = info["data_path"]
        output = []
        data_index_cache: dict[Path, Any] = {}
        for meta in rows:
            episode_index = int(meta["episode_index"])
            length = int(meta["length"])
            data_path = directory / _format_path(data_template,
                video_key=None, chunk=int(meta["data/chunk_index"]), file=int(meta["data/file_index"]))
            video_key = "observation.image"
            video_path = self._metadata_video_path(directory, info, meta)
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
            # Store the actual ffprobe presentation timestamps selected for this
            # episode. target_times are sensor-derived requests; matched_times are
            # the MP4 frames that passed the existing half-frame tolerance check.
            matched_times = video_times[nearest]
            output.append(_Episode(sensor, directory.name, episode_index, data_path, video_path,
                                   int(positions[0]), length, frames, timestamps, matched_times,
                                   nearest.astype(np.int64, copy=False), fps))
        return output

    def _video_times(self, path: Path) -> np.ndarray:
        if path not in self._video_timestamp_cache:
            progress = self._initialization_progress
            if progress is not None:
                progress.start_video(path)
            try:
                timestamps, cache_hit = self._timestamp_store.get(path)
            except BaseException:
                if progress is not None:
                    progress.fail_video()
                raise
            self._video_timestamp_cache[path] = timestamps
            if progress is not None:
                progress.finish_video(cache_hit=cache_hit)
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

    def _close_video(self) -> None:
        """Close the PyAV container owned by the current dataset process."""
        container = getattr(self, "_video", None)
        if container is not None:
            try:
                container.close()
            except Exception:
                # Worker shutdown must not mask the original exception.
                pass
        self._video = None
        self._video_stream = None
        self._video_path = None
        self._video_pid = None

    def _open_video(self, path: Path):
        """Return one process-local PyAV container and its first video stream."""
        process_id = os.getpid()
        if self._video_pid != process_id:
            # A dataset used before DataLoader fork can carry an inherited file
            # descriptor. Never reuse decoder state created by another process.
            self._close_video()
        if self._video_path != path:
            self._close_video()
            try:
                import av
            except ImportError as exc:
                raise ImportError("TaFDataset AV1 decoding requires PyAV (`uv add av`)") from exc
            try:
                container = av.open(str(path), mode="r")
                stream = container.streams.video[0]
                # PyAV's FFmpeg uses a software decoder for the TaF AV1 stream.
                # One decoder thread avoids native thread oversubscription when
                # several spawned DataLoader workers decode at the same time.
                stream.thread_type = "NONE"
                stream.codec_context.thread_count = 1
            except Exception:
                if "container" in locals():
                    container.close()
                raise
            self._video = container
            self._video_stream = stream
            self._video_path = path
            self._video_pid = process_id
        return self._video, self._video_stream

    @staticmethod
    def _timestamps_to_pts(timestamps: np.ndarray, time_base: Any) -> np.ndarray:
        """Convert ffprobe presentation times to exact integer stream PTS values."""
        seconds_per_tick = float(time_base)
        pts = np.rint(timestamps / seconds_per_tick).astype(np.int64)
        reconstructed = pts.astype(np.float64) * seconds_per_tick
        # ffprobe CSV timestamps have decimal formatting, so allow half a stream
        # tick while still requiring one unambiguous integer PTS per frame.
        if np.any(np.abs(reconstructed - timestamps) > seconds_per_tick / 2 + 1e-9):
            raise ValueError("ffprobe timestamps cannot be represented in the PyAV stream time base")
        return pts

    def _images(self, ep: _Episode, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode a window and close all FFmpeg state before returning.

        A DataLoader may start workers after the main process inspected a sample.
        Closing in finally ensures no live decoder or file descriptor reaches
        worker serialization or survives an exception.
        """
        try:
            return self._decode_images(ep, offsets)
        finally:
            self._close_video()

    def _decode_images(self, ep: _Episode, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode one temporal window from the preceding keyframe with PyAV.

        seek() receives the first requested PTS with backward=True and
        any_frame=False, so FFmpeg starts from a keyframe at or before it. Frames
        are then decoded once in presentation order; there is no per-frame seek.
        """
        container, stream = self._open_video(ep.video)
        expected_times = ep.video_timestamps[offsets]
        expected_pts = self._timestamps_to_pts(expected_times, stream.time_base)
        if np.any(np.diff(expected_pts) <= 0):
            raise ValueError(f"episode {ep.episode_index}: requested video PTS is not increasing")

        container.seek(
            int(expected_pts[0]),
            backward=True,
            any_frame=False,
            stream=stream,
        )
        images: list[np.ndarray] = []
        decoded_pts: list[int] = []
        wanted = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_pts = int(frame.pts)
            if current_pts < int(expected_pts[wanted]):
                continue
            if current_pts > int(expected_pts[wanted]):
                raise ValueError(
                    f"episode {ep.episode_index}: PyAV skipped expected PTS "
                    f"{int(expected_pts[wanted])} and reached {current_pts}"
                )

            if self.image_size is None:
                rgb = frame.to_ndarray(format="rgb24")
            else:
                size = ((self.image_size, self.image_size)
                        if isinstance(self.image_size, int) else self.image_size)
                resized = frame.reformat(
                    width=int(size[1]),
                    height=int(size[0]),
                    format="rgb24",
                    interpolation="AREA",
                )
                rgb = resized.to_ndarray()
            images.append(rgb)
            decoded_pts.append(current_pts)
            wanted += 1
            if wanted == len(expected_pts):
                break

        if wanted != len(expected_pts):
            raise ValueError(
                f"episode {ep.episode_index}: decoded {wanted}/{len(expected_pts)} "
                f"requested frames from {ep.video}"
            )
        actual_pts = np.asarray(decoded_pts, dtype=np.int64)
        if not np.array_equal(actual_pts, expected_pts):
            raise ValueError(f"episode {ep.episode_index}: ffprobe and PyAV PTS differ")
        return np.stack(images), actual_pts

    def __getitem__(self, index: int) -> dict[str, Any]:
        ep, start = self._locate(index)
        offsets = np.arange(start, start + self.window_size, dtype=np.int64)
        if offsets[-1] >= ep.length:
            raise RuntimeError("Window crossed an episode boundary")
        force, pressure = self._sensor_rows(ep, offsets)
        images, video_pts = self._images(ep, offsets)
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
                "video_frame_indices": torch.from_numpy(ep.video_frame_indices[offsets].copy()),
                "video_pts": torch.from_numpy(video_pts),
            },
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video"] = None
        state["_video_stream"] = None
        state["_video_path"] = None
        state["_video_pid"] = None
        state["_sensor_cache"] = None
        state["_video_timestamp_cache"] = {}
        return state

    def __del__(self) -> None:
        self._close_video()
