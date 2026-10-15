"""Console, JSONL, and optional Weights & Biases logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from utils.common import resolved_config


METRIC_KEYS = (
    "loss_total", "loss_nce", "loss_quant", "loss_codebook", "loss_commitment",
    "loss_recon", "loss_recon_pressure", "loss_recon_force",
    "positive_cosine", "negative_cosine", "recall_at_1", "recall_at_5",
    "pressure_codebook_usage", "force_codebook_usage",
    "pressure_codebook_usage_fraction", "force_codebook_usage_fraction",
    "pressure_codebook_perplexity", "force_codebook_perplexity",
    "pressure_dead_codes", "force_dead_codes",
)


def scalar_metrics(outputs: dict[str, Any]) -> dict[str, float]:
    return {key: float(outputs[key].detach()) for key in METRIC_KEYS}


class ExperimentLogger:
    def __init__(self, output_dir: Path, wandb_config: DictConfig,
                 resume_state: dict[str, Any] | None = None) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.config = wandb_config
        self.enabled = bool(wandb_config.enabled)
        self.resume_state = resume_state or {}
        self.run: Any | None = None
        self.last_wandb_step = int(self.resume_state.get("last_logged_step", -1))

    def start(self, full_config: DictConfig, model: torch.nn.Module) -> None:
        if not self.enabled:
            return
        try:
            import wandb

            run_id = self.resume_state.get("run_id")
            requested_name = self.config.name
            run_name = requested_name if requested_name is not None else self.resume_state.get("run_name")
            self.run = wandb.init(
                project=str(self.config.project),
                entity=self.config.entity,
                name=run_name,
                group=self.config.group,
                tags=list(self.config.tags),
                mode=str(self.config.mode),
                config=resolved_config(full_config),
                dir=str(self.output_dir),
                id=run_id,
                resume="allow" if run_id else None,
            )
            if self.run is None:
                raise RuntimeError("wandb.init() returned no run")
            self.run.define_metric("global_step")
            self.run.define_metric("train/*", step_metric="global_step")
            self.run.define_metric("val/*", step_metric="global_step")
            self.run.define_metric("epoch", step_metric="global_step")
            if bool(self.config.watch_model):
                self.run.watch(model, log="gradients", log_freq=int(self.config.log_every))
            print(
                f"W&B run: id={self.run.id}, name={self.run.name}, "
                f"mode={self.config.mode}, project={self.config.project}"
            )
        except Exception as exc:
            if self.run is not None:
                self.run.finish(exit_code=1)
                self.run = None
            hint = (
                " Configure WANDB_API_KEY or run `wandb login`."
                if str(self.config.mode) == "online" else ""
            )
            raise RuntimeError(
                f"W&B initialization failed in {self.config.mode!r} mode."
                f"{hint} No fallback mode was selected."
            ) from exc

    def write(self, record: dict[str, Any]) -> None:
        serialized = json.dumps(record, sort_keys=True)
        print(serialized)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")

    def log_wandb(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        if step <= self.last_wandb_step:
            raise ValueError(
                f"W&B step must increase monotonically: previous={self.last_wandb_step}, next={step}"
            )
        payload = {"global_step": step, **metrics}
        self.run.log(payload, step=step)
        self.last_wandb_step = step

    def wandb_state(self) -> dict[str, Any] | None:
        if self.run is None:
            return None
        return {
            "run_id": self.run.id,
            "run_name": self.run.name,
            "last_logged_step": self.last_wandb_step,
        }

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None
