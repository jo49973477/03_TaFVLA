"""Persistent ffprobe timestamp cache and initialization progress tests."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset.taf_dataset import (
    _InitializationProgress,
    _PersistentTimestampCache,
    TaFDataset,
    _default_timestamp_cache_dir,
    _validate_video_timestamps,
)


class TimestampCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "video.mp4"
        self.video.write_bytes(b"synthetic video bytes")
        self.cache_dir = self.root / "cache"
        self.timestamps = np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0], dtype=np.float64)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_first_probe_then_exact_cache_hit(self) -> None:
        cache = _PersistentTimestampCache(True, self.cache_dir)
        with patch(
            "dataset.taf_dataset._probe_video_timestamps",
            return_value=self.timestamps.copy(),
        ) as probe:
            first, first_hit = cache.get(self.video)
            second, second_hit = cache.get(self.video)

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(probe.call_count, 1)
        self.assertTrue(np.array_equal(first, self.timestamps))
        self.assertTrue(np.array_equal(second, self.timestamps))
        self.assertEqual(len(list(self.cache_dir.glob("*.npz"))), 1)

    def test_file_change_invalidates_only_that_video(self) -> None:
        cache = _PersistentTimestampCache(True, self.cache_dir)
        changed = np.asarray([0.0, 0.04, 0.08], dtype=np.float64)
        with patch(
            "dataset.taf_dataset._probe_video_timestamps",
            side_effect=[self.timestamps.copy(), changed.copy()],
        ) as probe:
            cache.get(self.video)
            self.video.write_bytes(b"changed size and content")
            os.utime(self.video, None)
            result, hit = cache.get(self.video)

        self.assertFalse(hit)
        self.assertEqual(probe.call_count, 2)
        self.assertTrue(np.array_equal(result, changed))

    def test_mtime_only_change_invalidates_cache(self) -> None:
        cache = _PersistentTimestampCache(True, self.cache_dir)
        changed = np.asarray([0.0, 0.05, 0.1], dtype=np.float64)
        with patch(
            "dataset.taf_dataset._probe_video_timestamps",
            side_effect=[self.timestamps.copy(), changed.copy()],
        ) as probe:
            cache.get(self.video)
            stat = self.video.stat()
            os.utime(
                self.video,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
            )
            result, hit = cache.get(self.video)

        self.assertFalse(hit)
        self.assertEqual(probe.call_count, 2)
        self.assertTrue(np.array_equal(result, changed))

    def test_corrupt_cache_is_regenerated_without_pickle(self) -> None:
        cache = _PersistentTimestampCache(True, self.cache_dir)
        with patch(
            "dataset.taf_dataset._probe_video_timestamps",
            return_value=self.timestamps.copy(),
        ) as probe:
            cache.get(self.video)
            cache_file = next(self.cache_dir.glob("*.npz"))
            cache_file.write_bytes(b"not an npz")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result, hit = cache.get(self.video)

        self.assertFalse(hit)
        self.assertEqual(probe.call_count, 2)
        self.assertTrue(np.array_equal(result, self.timestamps))
        self.assertTrue(any("regenerated" in str(item.message) for item in caught))

    def test_interrupted_run_keeps_completed_video_cache(self) -> None:
        second_video = self.root / "second.mp4"
        second_video.write_bytes(b"another video")
        cache = _PersistentTimestampCache(True, self.cache_dir)
        with patch(
            "dataset.taf_dataset._probe_video_timestamps",
            side_effect=[self.timestamps.copy(), KeyboardInterrupt()],
        ) as probe:
            cache.get(self.video)
            with self.assertRaises(KeyboardInterrupt):
                cache.get(second_video)
            result, hit = cache.get(self.video)

        self.assertTrue(hit)
        self.assertTrue(np.array_equal(result, self.timestamps))
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(len(list(self.cache_dir.glob("*.npz"))), 1)
        self.assertEqual(list(self.cache_dir.glob("*.tmp")), [])

    def test_concurrent_callers_publish_one_complete_cache(self) -> None:
        calls = 0
        calls_lock = threading.Lock()

        def slow_probe(_path: Path) -> np.ndarray:
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return self.timestamps.copy()

        first_cache = _PersistentTimestampCache(True, self.cache_dir)
        second_cache = _PersistentTimestampCache(True, self.cache_dir)
        with patch("dataset.taf_dataset._probe_video_timestamps", side_effect=slow_probe):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda cache: cache.get(self.video), [first_cache, second_cache]))

        self.assertEqual(calls, 1)
        self.assertEqual(sorted(hit for _, hit in results), [False, True])
        for values, _ in results:
            self.assertTrue(np.array_equal(values, self.timestamps))

    def test_cache_disabled_repeats_original_probe(self) -> None:
        cache = _PersistentTimestampCache(False, self.cache_dir)
        with patch(
            "dataset.taf_dataset._probe_video_timestamps",
            return_value=self.timestamps.copy(),
        ) as probe:
            cache.get(self.video)
            cache.get(self.video)
        self.assertEqual(probe.call_count, 2)
        self.assertFalse(self.cache_dir.exists())

    def test_xdg_default_and_timestamp_validation(self) -> None:
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root / "xdg")}, clear=False):
            self.assertEqual(
                _default_timestamp_cache_dir(),
                self.root / "xdg" / "taf-vla" / "video_timestamps",
            )
        invalid_arrays = [
            np.asarray([], dtype=np.float64),
            np.asarray([0.0, np.nan], dtype=np.float64),
            np.asarray([0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.1], dtype=np.float32),
            np.asarray([[0.0, 0.1]], dtype=np.float64),
        ]
        for values in invalid_arrays:
            with self.subTest(dtype=values.dtype, shape=values.shape):
                with self.assertRaises(ValueError):
                    _validate_video_timestamps(values, self.video)

    def test_non_tty_progress_has_sparse_plain_text_without_ansi(self) -> None:
        stream = io.StringIO()
        progress = _InitializationProgress(
            sequence_total=2,
            video_total=4,
            enabled=True,
            stream=stream,
            force_tty=False,
        )
        progress.start_sequence("sequence_a")
        progress.start_video(self.video)
        progress.finish_video(cache_hit=True)
        progress.finish_sequence()
        progress.start_video(self.video)
        progress.finish_video(cache_hit=False)
        progress.finish_sequence()
        progress.close()

        output = stream.getvalue()
        self.assertIn("TaF Dataset Initialization", output)
        self.assertIn("cache hits=1, misses=1, failed=0", output)
        self.assertNotIn("\x1b[", output)
        self.assertEqual(progress.stats["cache_hits"], 1)
        self.assertEqual(progress.stats["cache_misses"], 1)

    def test_missing_episode_metadata_is_reconstructed_in_memory(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        sequence = self.root / "dataset" / "taf_dataset" / "synthetic" / "recovered_obj1"
        data_path = sequence / "data" / "chunk-000" / "file-000.parquet"
        video_path = sequence / "videos" / "observation.image" / "chunk-000" / "file-000.mp4"
        data_path.parent.mkdir(parents=True)
        video_path.parent.mkdir(parents=True)
        (sequence / "meta").mkdir(parents=True)
        frame_count = 6
        pq.write_table(
            pa.table({
                "index": np.arange(frame_count, dtype=np.int64),
                "episode_index": np.zeros(frame_count, dtype=np.int64),
                "frame_index": np.arange(frame_count, dtype=np.int64),
                "timestamp": np.arange(frame_count, dtype=np.float64) / 30.0,
                "observation.force_torque": pa.array(
                    np.zeros((frame_count, 6), dtype=np.float32).tolist(),
                    type=pa.list_(pa.float32(), 6),
                ),
                "observation.pressure_matrix": pa.array(
                    np.zeros((frame_count, 144), dtype=np.float32).tolist(),
                    type=pa.list_(pa.float32(), 144),
                ),
            }),
            data_path,
        )
        video_path.write_bytes(b"video placeholder")
        info = {
            "total_episodes": 1,
            "total_frames": frame_count,
            "fps": 30,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": (
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            ),
        }
        (sequence / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

        with patch(
            "dataset.taf_dataset._probe_video_timestamps",
            return_value=np.arange(frame_count, dtype=np.float64) / 30.0,
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset = TaFDataset(
                self.root / "dataset",
                window_size=5,
                timestamp_cache_enabled=False,
                show_initialization_progress=False,
            )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.episode_count, 1)
        self.assertEqual(dataset.reconstructed_episode_metadata, ["recovered_obj1"])
        self.assertFalse((sequence / "meta" / "episodes").exists())
        self.assertTrue(any("reconstructed episode boundaries" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
