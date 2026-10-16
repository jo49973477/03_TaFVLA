"""Fast tests for fixed-subset cross-modal retrieval and visualization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.alignment import (
    alignment_schedule,
    collect_alignment_embeddings,
    compute_alignment_metrics,
    create_alignment_visualizations,
    joint_umap_projection,
    matching_segments,
    save_alignment_selection,
    select_alignment_indices,
    should_run_final_alignment,
    similarity_distributions,
)


class _WindowDataset:
    window_size = 5

    def __init__(self) -> None:
        self.episodes = [
            SimpleNamespace(sequence_id="train", episode_index=0),
            SimpleNamespace(sequence_id="val_a", episode_index=0),
            SimpleNamespace(sequence_id="val_b", episode_index=0),
        ]
        self._ends = [12, 24, 36]

    def _locate(self, index: int):
        previous = 0
        for episode, end in zip(self.episodes, self._ends, strict=True):
            if index < end:
                return episode, index - previous
            previous = end
        raise IndexError(index)


class AlignmentTests(unittest.TestCase):
    def test_embedding_collection_uses_eval_and_inference_mode(self) -> None:
        test_case = self

        class ProbeModel(torch.nn.Module):
            def forward(self, *, tactile, force, pressure):
                test_case.assertFalse(self.training)
                test_case.assertFalse(torch.is_grad_enabled())
                return {
                    "z_tactile": tactile[:, 0, :, 0, 0],
                    "c_force": force[:, 0, :3],
                }

        samples = [
            {
                "tactile": torch.full((1, 3, 1, 1), float(index)),
                "force": torch.full((1, 6), float(index)),
                "pressure": torch.zeros(1, 12, 12),
            }
            for index in range(4)
        ]
        model = ProbeModel().train()
        tactile, force = collect_alignment_embeddings(
            model,
            DataLoader(samples, batch_size=2, shuffle=False),
            torch.device("cpu"),
        )
        self.assertTrue(model.training)
        self.assertEqual(tuple(tactile.shape), (4, 3))
        self.assertTrue(torch.equal(tactile, force))

    def test_perfect_alignment_has_perfect_bidirectional_retrieval(self) -> None:
        embeddings = torch.eye(12)
        result = compute_alignment_metrics(embeddings, embeddings.clone())
        self.assertTrue(torch.allclose(result.similarity, torch.eye(12)))
        for direction in ("t2f", "f2t"):
            self.assertEqual(result.metrics[f"alignment/{direction}/recall_at_1"], 1.0)
            self.assertEqual(result.metrics[f"alignment/{direction}/recall_at_5"], 1.0)
            self.assertEqual(result.metrics[f"alignment/{direction}/mrr"], 1.0)
            self.assertAlmostEqual(
                result.metrics[f"alignment/{direction}/positive_cosine"], 1.0
            )

    def test_random_embeddings_are_near_random_retrieval_baseline(self) -> None:
        generator = torch.Generator().manual_seed(7)
        tactile = torch.randn(512, 64, generator=generator)
        force = torch.randn(512, 64, generator=generator)
        metrics = compute_alignment_metrics(tactile, force).metrics
        self.assertLess(abs(metrics["alignment/t2f/recall_at_1"] - 1 / 512), 0.02)
        self.assertLess(abs(metrics["alignment/t2f/recall_at_5"] - 5 / 512), 0.04)
        self.assertLess(metrics["alignment/t2f/mrr"], 0.05)

    def test_similarity_distribution_values_are_exact(self) -> None:
        similarity = torch.tensor([
            [0.9, 0.1, 0.2],
            [0.3, 0.8, 0.4],
            [-0.1, 0.0, 0.7],
        ])
        values = similarity_distributions(similarity, random_seed=3)
        np.testing.assert_allclose(values["positive"], [0.9, 0.8, 0.7])
        np.testing.assert_allclose(values["negative"], [0.1, 0.2, 0.3, 0.4, -0.1, 0.0])
        np.testing.assert_allclose(values["hard_negative"], [0.2, 0.4, 0.0])

    def test_fixed_validation_selection_has_no_leakage_or_overlap(self) -> None:
        dataset = _WindowDataset()
        training = list(range(12))
        validation = list(range(12, 36))
        first = select_alignment_indices(
            dataset,
            validation,
            max_pairs=6,
            min_window_start_gap=3,
            random_seed=42,
            training_indices=training,
        )
        second = select_alignment_indices(
            dataset,
            validation,
            max_pairs=6,
            min_window_start_gap=3,
            random_seed=42,
            training_indices=training,
        )
        self.assertEqual(first.indices, second.indices)
        self.assertTrue(set(first.indices).issubset(validation))
        self.assertTrue(set(first.indices).isdisjoint(training))
        self.assertEqual(first.effective_min_gap, dataset.window_size)
        by_episode: dict[tuple[str, int], list[int]] = {}
        for sample in first.samples:
            key = (sample["sequence_id"], sample["episode_index"])
            by_episode.setdefault(key, []).append(sample["window_start"])
        for starts in by_episode.values():
            for left_index, left in enumerate(starts):
                for right in starts[left_index + 1:]:
                    self.assertGreaterEqual(abs(left - right), dataset.window_size)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            save_alignment_selection(first, path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["indices"], first.indices)
            self.assertEqual(saved["random_seed"], 42)

        with self.assertRaisesRegex(ValueError, "leaked"):
            select_alignment_indices(
                dataset,
                validation + [0],
                max_pairs=4,
                min_window_start_gap=5,
                random_seed=42,
                training_indices=training,
            )

    def test_joint_umap_fits_combined_modalities_once_and_pairs_by_index(self) -> None:
        calls = []

        class FakeReducer:
            def fit_transform(self, values):
                calls.append(values.copy())
                return np.arange(len(values) * 2, dtype=np.float32).reshape(-1, 2)

        fake_umap = SimpleNamespace(UMAP=lambda **kwargs: FakeReducer())
        tactile = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        force = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.8]])
        with patch.dict(sys.modules, {"umap": fake_umap}):
            tactile_2d, force_2d = joint_umap_projection(
                tactile, force, random_seed=11,
            )
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0][:3], tactile.numpy())
        np.testing.assert_allclose(calls[0][3:], force.numpy())
        segments = matching_segments(tactile_2d, force_2d)
        np.testing.assert_allclose(segments[:, 0], tactile_2d)
        np.testing.assert_allclose(segments[:, 1], force_2d)

    def test_visualizations_save_three_pngs_and_close_figures(self) -> None:
        embeddings = torch.eye(8)
        result = compute_alignment_metrics(embeddings, embeddings)
        tactile_2d = np.column_stack((np.arange(8), np.zeros(8))).astype(np.float32)
        force_2d = tactile_2d + np.array([0.1, 0.2], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory, patch(
            "utils.alignment.joint_umap_projection",
            return_value=(tactile_2d, force_2d),
        ):
            images, paths = create_alignment_visualizations(
                result,
                max_pairs=8,
                random_seed=42,
                title_label="Epoch 5",
                file_prefix="epoch_0005",
                output_dir=Path(directory),
                save_local_figures=True,
            )
            self.assertEqual(len(images), 3)
            self.assertEqual(set(paths), {"similarity", "umap", "distribution"})
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            self.assertTrue(all(image.ndim == 3 and image.shape[2] == 3 for image in images.values()))
            self.assertEqual(plt.get_fignums(), [])

    def test_epoch_and_final_triggers_handle_mid_epoch_stop(self) -> None:
        self.assertEqual(
            alignment_schedule(
                5,
                retrieval_every_n_epochs=2,
                visualize_every_n_epochs=5,
            ),
            (True, True),
        )
        self.assertEqual(
            alignment_schedule(
                4,
                retrieval_every_n_epochs=2,
                visualize_every_n_epochs=5,
            ),
            (True, False),
        )
        self.assertTrue(should_run_final_alignment(
            enabled=True,
            run_on_training_end=True,
            current_step=13,
            last_evaluated_step=10,
        ))
        self.assertFalse(should_run_final_alignment(
            enabled=True,
            run_on_training_end=True,
            current_step=13,
            last_evaluated_step=13,
        ))


if __name__ == "__main__":
    unittest.main()
