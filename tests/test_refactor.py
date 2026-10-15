"""Fast smoke tests for config, data utilities, logging, and checkpoints."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from model.taf_adapter import TaFAdapter
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.common import validate_config
from utils.config import register_config_schema
from utils.data import TemporalDiverseBatchSampler, split_by_group
from utils.logging import ExperimentLogger
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

        invalid = compose_config("data.window_size=0")
        with self.assertRaisesRegex(ValueError, "data.window_size"):
            validate_config(invalid)

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
            )
            checkpoint = load_checkpoint(checkpoint_path)
            self.assertEqual(checkpoint["step"], 12)
            self.assertEqual(checkpoint["wandb"]["run_id"], "offline-id")

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
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentLogger(Path(directory), config.wandb)
            logger.start(config, torch.nn.Linear(1, 1))
            logger.write({"step": 1, "train/loss_total": 2.0})
            logger.finish()
            lines = (Path(directory) / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(lines[0])["train/loss_total"], 2.0)

    def test_offline_wandb_logs_without_network(self) -> None:
        config = compose_config("wandb.enabled=true", "wandb.mode=offline")
        with tempfile.TemporaryDirectory() as directory:
            logger = ExperimentLogger(Path(directory), config.wandb)
            logger.start(config, torch.nn.Linear(1, 1))
            logger.write({"step": 1, "train/loss_total": 2.0})
            logger.log_wandb({"train/loss_total": 2.0, "val/loss_total": 1.5}, step=1)
            state = logger.wandb_state()
            logger.finish()
            self.assertIsNotNone(state)
            self.assertTrue(state["run_id"])
            self.assertEqual(state["last_logged_step"], 1)
            self.assertTrue(any((Path(directory) / "wandb").glob("offline-run-*")))
            self.assertTrue((Path(directory) / "metrics.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
