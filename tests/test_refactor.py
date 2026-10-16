"""Fast smoke tests for config, data utilities, logging, and checkpoints."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from model.taf_adapter import TaFAdapter
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.common import validate_config
from utils.config import register_config_schema
from utils.data import TemporalDiverseBatchSampler, split_by_group
from utils.logging import ExperimentLogger, TrainingProgress
from utils.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]


def compose_config(*overrides: str):
    register_config_schema()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        return compose(config_name="taf_adapter", overrides=list(overrides))


class _FakeDataset:
    def __init__(self) -> None:
        self.episodes = [
            SimpleNamespace(sequence_id="a", episode_index=0),
            SimpleNamespace(sequence_id="a", episode_index=1),
            SimpleNamespace(sequence_id="b", episode_index=0),
        ]
        self._ends = [3, 6, 10]

    def _locate(self, index: int):
        previous = 0
        for episode, end in zip(self.episodes, self._ends, strict=True):
            if index < end:
                return episode, index - previous
            previous = end
        raise IndexError(index)


class RefactorSmokeTests(unittest.TestCase):
    def test_hydra_override_and_semantic_validation(self) -> None:
        config = compose_config(
            "data.window_size=10", "training.batch_size=16",
            "training.learning_rate=1e-4", "loss.temperature=0.07",
        )
        validate_config(config)
        self.assertEqual(config.data.window_size, 10)
        self.assertEqual(config.training.batch_size, 16)
        self.assertNotIn("commitment_beta", config.loss)
        self.assertTrue(config.alignment_eval.enabled)
        self.assertTrue(config.data.timestamp_cache_enabled)
        self.assertIsNone(config.data.timestamp_cache_dir)
        self.assertTrue(config.data.show_initialization_progress)

        alignment = compose_config(
            "alignment_eval.visualize_every_n_epochs=10",
            "alignment_eval.retrieval_max_pairs=128",
            "alignment_eval.visualization_max_pairs=32",
        )
        validate_config(alignment)
        self.assertEqual(alignment.alignment_eval.visualize_every_n_epochs, 10)

        invalid = compose_config("data.window_size=0")
        with self.assertRaisesRegex(ValueError, "data.window_size"):
            validate_config(invalid)

        invalid_alignment = compose_config(
            "alignment_eval.retrieval_max_pairs=4",
            "alignment_eval.visualization_max_pairs=8",
        )
        with self.assertRaisesRegex(ValueError, "visualization_max_pairs"):
            validate_config(invalid_alignment)

        cache_override = compose_config(
            "data.timestamp_cache_enabled=false",
            "data.timestamp_cache_dir=/tmp/taf-cache",
            "data.show_initialization_progress=false",
        )
        validate_config(cache_override)
        self.assertFalse(cache_override.data.timestamp_cache_enabled)
        self.assertEqual(cache_override.data.timestamp_cache_dir, "/tmp/taf-cache")

    def test_hydra_rejects_unknown_key_and_bad_type(self) -> None:
        with self.assertRaises(ConfigCompositionException):
            compose_config("training.learnng_rate=1e-4")
        with self.assertRaises(ConfigCompositionException):
            compose_config("data.window_size=wrong")

    def test_group_split_has_no_sequence_leakage(self) -> None:
        dataset = _FakeDataset()
        train, validation, train_episodes, val_episodes, train_keys, val_keys = split_by_group(
            dataset, unit="sequence", validation_fraction=0.5, seed=42,
        )
        self.assertTrue(train and validation)
        self.assertTrue(train_episodes.isdisjoint(val_episodes))
        self.assertTrue(set(train_keys).isdisjoint(val_keys))
        self.assertEqual(set(train + validation), set(range(10)))

    def test_temporal_sampler_is_deterministic_and_complete(self) -> None:
        dataset = _FakeDataset()
        indices = list(range(10))
        first = list(TemporalDiverseBatchSampler(dataset, indices, 2, 2, 7))
        second = list(TemporalDiverseBatchSampler(dataset, indices, 2, 2, 7))
        self.assertEqual(first, second)
        self.assertEqual(sorted(index for batch in first for index in batch), indices)

    def test_checkpoint_roundtrip_and_legacy_load(self) -> None:
        config = compose_config()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        normalization = {
            "pressure_mean": torch.zeros(12, 12),
            "pressure_std": torch.ones(12, 12),
            "force_mean": torch.zeros(6),
            "force_std": torch.ones(6),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "current.pt"
            save_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, scaler=scaler,
                step=12, epoch=3, config=config, normalization=normalization,
                train_keys=["a"], val_keys=["b"],
                wandb_state={"run_id": "offline-id", "last_logged_step": 12},
                alignment_state={"completed_epochs": 2, "sample_indices": [4, 8]},
            )
            checkpoint = load_checkpoint(checkpoint_path)
            self.assertEqual(checkpoint["step"], 12)
            self.assertEqual(checkpoint["wandb"]["run_id"], "offline-id")
            self.assertEqual(checkpoint["alignment"]["completed_epochs"], 2)

            legacy_path = Path(directory) / "legacy.pt"
            torch.save({"model": model.state_dict(), "step": 4}, legacy_path)
            legacy = load_checkpoint(legacy_path)
            self.assertEqual(legacy["step"], 4)
            self.assertNotIn("wandb", legacy)

    def test_canonical_beta_preserves_legacy_forward(self) -> None:
        config = compose_config("tactile_encoder.pretrained=false")
        canonical = build_model(config).eval()
        sections = OmegaConf.to_container(config, resolve=True)
        legacy_loss = dict(sections["loss"])
        legacy_loss["commitment_beta"] = sections["vq"]["commitment_beta"]
        legacy = TaFAdapter(
            window_size=sections["data"]["window_size"],
            force_encoder=sections["force_encoder"],
            vq=sections["vq"],
            tactile_encoder=sections["tactile_encoder"],
            loss=legacy_loss,
            normalization_eps=sections["data"]["normalization_eps"],
        ).eval()
        legacy.load_state_dict(canonical.state_dict(), strict=True)

        torch.manual_seed(17)
        batch, window = 2, int(config.data.window_size)
        features = torch.randn(batch, window, int(config.tactile_encoder.transformer_dim))
        force = torch.randn(batch, window, 6)
        pressure = torch.randn(batch, window, 12, 12)
        with torch.no_grad():
            current_output = canonical.forward_from_tactile_features(features, force, pressure)
            legacy_output = legacy.forward_from_tactile_features(features, force, pressure)
        for key in ("z_tactile", "c_force", "loss_total", "loss_quant", "loss_recon"):
            self.assertTrue(torch.equal(current_output[key], legacy_output[key]), key)

    def test_disabled_logger_keeps_jsonl(self) -> None:
        config = compose_config("wandb.enabled=false")
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentLogger(
                Path(directory), config.wandb,
                console_stream=console, force_tty=False,
            )
            logger.start(config, torch.nn.Linear(1, 1))
            logger.start_progress(max_steps=2)
            record = {
                "step": 1,
                "train/loss_total": 2.0,
                "train/loss_nce": 1.0,
                "train/loss_quant": 0.5,
                "train/loss_recon": 0.5,
                "train/pressure_codebook_perplexity": 8.0,
                "train/force_codebook_perplexity": 7.0,
                "train/pressure_codebook_usage": 4.0,
                "train/pressure_dead_codes": 12.0,
            }
            logger.advance(1)
            logger.write(record)
            logger.update_training(record)
            logger.finish()
            lines = (Path(directory) / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(lines[0])["train/loss_total"], 2.0)
            self.assertEqual(json.loads(lines[0])["train/pressure_dead_codes"], 12.0)
            self.assertIn("Train step 1/2", console.getvalue())
            self.assertNotIn('{"step"', console.getvalue())

    def test_wandb_filter_and_resume_step_are_independent_from_jsonl(self) -> None:
        class FakeRun:
            id = "run-id"
            name = "run-name"

            def __init__(self) -> None:
                self.logged = []

            def log(self, payload, step) -> None:
                self.logged.append((payload, step))

            def finish(self, exit_code=0) -> None:
                pass

        config = compose_config("wandb.enabled=true", "wandb.mode=offline")
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentLogger(
                Path(directory), config.wandb,
                resume_state={"last_logged_step": 4},
                console_stream=console, force_tty=False,
            )
            fake_run = FakeRun()
            logger.run = fake_run
            logger.start_progress(max_steps=10, initial_step=4)
            logger.advance(5)
            metrics = {
                "train/loss_total": 2.0,
                "train/loss_nce": 1.0,
                "train/loss_quant": 0.5,
                "train/loss_recon": 0.5,
                "val/loss_total": 1.8,
                "val/loss_nce": 0.9,
                "val/loss_quant": 0.4,
                "val/loss_recon": 0.5,
                "train/pressure_codebook_perplexity": 8.0,
                "train/force_codebook_perplexity": 7.0,
                "val/pressure_codebook_perplexity": 6.0,
                "val/force_codebook_perplexity": 5.0,
                "train/pressure_codebook_usage": 4.0,
                "train/force_codebook_usage": 3.0,
                "val/pressure_codebook_usage": 4.0,
                "val/force_codebook_usage": 3.0,
                "train/pressure_codebook_usage_fraction": 0.25,
                "train/force_codebook_usage_fraction": 0.25,
                "val/pressure_codebook_usage_fraction": 0.25,
                "val/force_codebook_usage_fraction": 0.25,
                "train/pressure_dead_codes": 12.0,
                "train/force_dead_codes": 13.0,
                "val/pressure_dead_codes": 12.0,
                "val/force_dead_codes": 13.0,
                "val/recall_at_1": 0.75,
            }
            logger.write({"step": 5, **metrics})
            logger.log_wandb(metrics, step=5)

            payload, logged_step = fake_run.logged[0]
            self.assertEqual(logged_step, 5)
            self.assertEqual(payload["global_step"], 5)
            for prefix in ("train", "val"):
                for name in ("loss_total", "loss_nce", "loss_quant", "loss_recon"):
                    self.assertIn(f"{prefix}/{name}", payload)
            self.assertIn("train/pressure_codebook_perplexity", payload)
            self.assertIn("val/force_codebook_perplexity", payload)
            self.assertIn("val/recall_at_1", payload)
            excluded_names = (
                "pressure_codebook_usage", "force_codebook_usage",
                "pressure_codebook_usage_fraction", "force_codebook_usage_fraction",
                "pressure_dead_codes", "force_dead_codes",
            )
            for prefix in ("train", "val"):
                for name in excluded_names:
                    self.assertNotIn(f"{prefix}/{name}", payload)
            with self.assertRaisesRegex(ValueError, "must increase monotonically"):
                logger.log_wandb({"train/loss_total": 1.0}, step=5)
            logger.log_wandb(
                {"completed_epoch": 1, "alignment/t2f/recall_at_1": 0.8},
                step=5,
                allow_same_step=True,
            )
            self.assertEqual(fake_run.logged[-1][1], 5)
            self.assertEqual(
                fake_run.logged[-1][0]["alignment/t2f/recall_at_1"], 0.8
            )

            saved = json.loads((Path(directory) / "metrics.jsonl").read_text())
            self.assertEqual(saved["train/pressure_codebook_usage"], 4.0)
            self.assertEqual(saved["val/force_dead_codes"], 13.0)
            self.assertEqual(logger.progress.current_step, 5)
            logger.finish()

    def test_tty_dashboard_uses_fixed_positions_and_refreshes_in_place(self) -> None:
        created = []

        class FakeTqdm:
            def __init__(self, *args, **kwargs) -> None:
                self.position = kwargs["position"]
                self.n = kwargs.get("initial", 0)
                self.description = kwargs.get("desc", "")
                self.refresh_count = 0
                self.closed = False
                created.append(self)

            def update(self, amount) -> None:
                self.n += amount

            def set_description_str(self, description, refresh=False) -> None:
                self.description = description

            def refresh(self) -> None:
                self.refresh_count += 1

            def close(self) -> None:
                self.closed = True

            @staticmethod
            def write(message, file=None) -> None:
                file.write(message + "\n")

        with patch("utils.logging.tqdm", FakeTqdm):
            progress = TrainingProgress(
                max_steps=10,
                initial_step=3,
                stream=io.StringIO(),
                force_tty=True,
            )
            self.assertEqual([bar.position for bar in created], list(range(9)))
            progress.advance(4)
            progress.update_training({
                "step": 4,
                "train/loss_total": 2.0,
                "train/loss_nce": 1.0,
                "train/loss_quant": 0.5,
                "train/loss_recon": 0.5,
                "train/pressure_codebook_perplexity": 8.0,
                "train/force_codebook_perplexity": 7.0,
            })
            progress.update_validation({
                "step": 4,
                "val/loss_total": 1.8,
                "val/loss_nce": 0.9,
                "val/loss_quant": 0.4,
                "val/loss_recon": 0.5,
                "val/recall_at_1": 0.75,
            })
            progress.update_alignment({
                "evaluation_label": "epoch",
                "completed_epoch": 2,
                "global_step": 4,
                "alignment/t2f/recall_at_1": 0.75,
                "alignment/t2f/recall_at_5": 1.0,
                "alignment/t2f/mrr": 0.8,
                "alignment/t2f/positive_cosine": 0.7,
                "alignment/t2f/negative_cosine": 0.1,
                "alignment/t2f/hard_negative_margin": 0.2,
                "alignment/f2t/recall_at_1": 0.7,
                "alignment/f2t/recall_at_5": 0.9,
                "alignment/f2t/mrr": 0.75,
            })
            self.assertEqual(len(created), 9)
            self.assertEqual([bar.position for bar in created], list(range(9)))
            self.assertEqual(created[0].n, 4)
            self.assertIn("Pressure PPL", created[2].description)
            self.assertIn("Val @    4", created[3].description)
            self.assertIn("Recall@1", created[4].description)
            self.assertIn("Epoch 2", created[5].description)
            self.assertIn("Tactile -> Force", created[6].description)
            self.assertIn("Hard Gap", created[7].description)
            self.assertIn("Force -> Tactile", created[8].description)
            progress.close()
            self.assertTrue(all(bar.closed for bar in created))

    def test_offline_wandb_logs_without_network(self) -> None:
        config = compose_config("wandb.enabled=true", "wandb.mode=offline")
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentLogger(Path(directory), config.wandb)
            logger.start(config, torch.nn.Linear(1, 1))
            logger.write({"step": 1, "train/loss_total": 2.0})
            logger.log_wandb({"train/loss_total": 2.0, "val/loss_total": 1.5}, step=1)
            logger.log_wandb(
                {"completed_epoch": 1, "alignment/t2f/recall_at_1": 0.75},
                step=1,
                allow_same_step=True,
                images={
                    "alignment/visualization/similarity_matrix":
                        torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
                },
            )
            state = logger.wandb_state()
            logger.finish()
            self.assertIsNotNone(state)
            self.assertTrue(state["run_id"])
            self.assertEqual(state["last_logged_step"], 1)
            self.assertTrue(any((Path(directory) / "wandb").glob("offline-run-*")))
            self.assertTrue((Path(directory) / "metrics.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
