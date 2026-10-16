"""JSONL, Weights & Biases, and terminal progress logging."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

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

# 모델은 아래 값을 계속 계산하고 JSONL에도 모두 보존한다. W&B 대시보드만
# 대표 다양성 지표인 perplexity 두 개로 단순화한다.
WANDB_EXCLUDED_CODEBOOK_METRICS = frozenset({
    "pressure_codebook_usage",
    "force_codebook_usage",
    "pressure_codebook_usage_fraction",
    "force_codebook_usage_fraction",
    "pressure_dead_codes",
    "force_dead_codes",
})


def scalar_metrics(outputs: dict[str, Any]) -> dict[str, float]:
    """모델 출력 Tensor를 파일과 화면에 기록할 Python scalar로 변환한다."""
    return {key: float(outputs[key].detach()) for key in METRIC_KEYS}


def filter_wandb_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """W&B에서만 상세 codebook 사용량 지표를 제외한다.

    ``train/pressure_codebook_usage``처럼 namespace가 붙은 키와 namespace가
    없는 키를 모두 처리한다. 입력 dictionary는 수정하지 않으므로 같은 값이
    JSONL 기록이나 checkpoint 상태에 사용되어도 영향이 없다.
    """
    return {
        key: value
        for key, value in metrics.items()
        if key.rsplit("/", maxsplit=1)[-1] not in WANDB_EXCLUDED_CODEBOOK_METRICS
    }


class TrainingProgress:
    """Optimizer step 진행률과 고정 위치 metric dashboard를 관리한다.

    TTY에서는 tqdm의 position 0~8을 고정해 진행률, loss, retrieval 행만 갱신한다. 파이프,
    CI 로그처럼 TTY가 아닌 출력에서는 ANSI 제어 문자를 쓰지 않고 logger가
    metric을 전달하는 주기에만 한 줄 요약을 출력한다.
    """

    def __init__(
        self,
        max_steps: int,
        initial_step: int = 0,
        *,
        stream: TextIO | None = None,
        force_tty: bool | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        detected_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.dynamic = detected_tty if force_tty is None else force_tty
        self.max_steps = max_steps
        self.current_step = min(max(initial_step, 0), max_steps)
        self.train_metrics: dict[str, Any] = {}
        self.val_metrics: dict[str, Any] = {}
        self.alignment_metrics: dict[str, Any] = {}
        self.closed = False
        self.progress_bar: Any | None = None
        self.dashboard_lines: list[Any] = []

        if self.dynamic:
            self._start_dynamic_ui()

    def _start_dynamic_ui(self) -> None:
        """Progress bar 한 줄과 dashboard 여덟 줄을 고정 position에 만든다."""
        self.progress_bar = tqdm(
            total=self.max_steps,
            initial=self.current_step,
            desc="Optimizer steps",
            unit="step",
            position=0,
            leave=True,
            dynamic_ncols=True,
            file=self.stream,
            bar_format=(
                "{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}% "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
        for position in range(1, 9):
            line = tqdm(
                total=0,
                position=position,
                leave=True,
                file=self.stream,
                bar_format="{desc}",
            )
            self.dashboard_lines.append(line)
        self._refresh_dashboard()

    @staticmethod
    def _value(metrics: dict[str, Any], key: str) -> str:
        """누락된 값은 고정 폭 placeholder로, scalar는 소수점 네 자리로 표시한다."""
        value = metrics.get(key)
        if value is None:
            return "   --   "
        return f"{float(value):8.4f}"

    def _dashboard_text(self) -> list[str]:
        train = self.train_metrics
        val = self.val_metrics
        alignment = self.alignment_metrics
        val_step = val.get("step")
        val_step_text = "----" if val_step is None else f"{int(val_step):>4d}"
        completed_epoch = int(alignment.get("completed_epoch", 0))
        evaluation_label = alignment.get("evaluation_label", "epoch")
        if not alignment:
            alignment_title = "Waiting"
        elif evaluation_label == "final" and completed_epoch == 0:
            alignment_title = "Final (no completed epoch)"
        elif evaluation_label == "final":
            alignment_title = f"Final after Epoch {completed_epoch}"
        else:
            alignment_title = f"Epoch {completed_epoch}"
        return [
            (
                "Train        | Total " + self._value(train, "train/loss_total")
                + " | NCE " + self._value(train, "train/loss_nce")
                + " | Quant " + self._value(train, "train/loss_quant")
                + " | Recon " + self._value(train, "train/loss_recon")
            ),
            (
                "             | Pressure PPL "
                + self._value(train, "train/pressure_codebook_perplexity")
                + " | Force PPL "
                + self._value(train, "train/force_codebook_perplexity")
            ),
            (
                f"Val @ {val_step_text}   | Total " + self._value(val, "val/loss_total")
                + " | NCE " + self._value(val, "val/loss_nce")
                + " | Quant " + self._value(val, "val/loss_quant")
                + " | Recon " + self._value(val, "val/loss_recon")
            ),
            "             | Recall@1 " + self._value(val, "val/recall_at_1"),
            f"Cross-Modal Retrieval | {alignment_title}",
            (
                "Tactile -> Force | R@1 "
                + self._value(alignment, "alignment/t2f/recall_at_1")
                + " | R@5 " + self._value(alignment, "alignment/t2f/recall_at_5")
                + " | MRR " + self._value(alignment, "alignment/t2f/mrr")
            ),
            (
                "                  | Positive "
                + self._value(alignment, "alignment/t2f/positive_cosine")
                + " | Negative " + self._value(alignment, "alignment/t2f/negative_cosine")
                + " | Hard Gap "
                + self._value(alignment, "alignment/t2f/hard_negative_margin")
            ),
            (
                "Force -> Tactile | R@1 "
                + self._value(alignment, "alignment/f2t/recall_at_1")
                + " | R@5 " + self._value(alignment, "alignment/f2t/recall_at_5")
                + " | MRR " + self._value(alignment, "alignment/f2t/mrr")
            ),
        ]

    def _refresh_dashboard(self) -> None:
        if not self.dynamic or self.closed:
            return
        for line, description in zip(
            self.dashboard_lines, self._dashboard_text(), strict=True
        ):
            line.set_description_str(description, refresh=False)
            line.refresh()

    def advance(self, step: int) -> None:
        """실제 optimizer step까지 진행률을 이동한다 (micro-step은 받지 않는다)."""
        if step < self.current_step:
            raise ValueError(
                f"Progress step cannot decrease: previous={self.current_step}, next={step}"
            )
        increment = step - self.current_step
        self.current_step = step
        if self.dynamic and self.progress_bar is not None and increment:
            self.progress_bar.update(increment)

    def update_training(self, metrics: dict[str, Any]) -> None:
        """최근 training 집계값으로 dashboard 또는 non-TTY 요약을 갱신한다."""
        self.train_metrics = dict(metrics)
        if self.dynamic:
            self._refresh_dashboard()
            return
        self.message(
            f"Train step {int(metrics['step'])}/{self.max_steps} | "
            f"total={self._value(metrics, 'train/loss_total').strip()} "
            f"nce={self._value(metrics, 'train/loss_nce').strip()} "
            f"quant={self._value(metrics, 'train/loss_quant').strip()} "
            f"recon={self._value(metrics, 'train/loss_recon').strip()} "
            f"pressure_ppl={self._value(metrics, 'train/pressure_codebook_perplexity').strip()} "
            f"force_ppl={self._value(metrics, 'train/force_codebook_perplexity').strip()}"
        )

    def update_validation(self, metrics: dict[str, Any]) -> None:
        """최근 validation 결과를 보존하고 dashboard의 고정된 두 줄을 갱신한다."""
        self.val_metrics = dict(metrics)
        if self.dynamic:
            self._refresh_dashboard()
            return
        self.message(
            f"Val step {int(metrics['step'])} | "
            f"total={self._value(metrics, 'val/loss_total').strip()} "
            f"nce={self._value(metrics, 'val/loss_nce').strip()} "
            f"quant={self._value(metrics, 'val/loss_quant').strip()} "
            f"recon={self._value(metrics, 'val/loss_recon').strip()} "
            f"recall@1={self._value(metrics, 'val/recall_at_1').strip()}"
        )

    def update_alignment(self, metrics: dict[str, Any]) -> None:
        """최근 epoch/final retrieval 결과를 고정 dashboard에 반영한다."""
        self.alignment_metrics = dict(metrics)
        if self.dynamic:
            self._refresh_dashboard()
            return
        self.message(
            f"Alignment {metrics['evaluation_label']} "
            f"completed_epoch={int(metrics['completed_epoch'])} "
            f"step={int(metrics['global_step'])} | "
            f"t2f_r@1={self._value(metrics, 'alignment/t2f/recall_at_1').strip()} "
            f"r@5={self._value(metrics, 'alignment/t2f/recall_at_5').strip()} "
            f"mrr={self._value(metrics, 'alignment/t2f/mrr').strip()} "
            f"positive={self._value(metrics, 'alignment/t2f/positive_cosine').strip()} "
            f"negative={self._value(metrics, 'alignment/t2f/negative_cosine').strip()} "
            f"hard_gap={self._value(metrics, 'alignment/t2f/hard_negative_margin').strip()}"
        )

    def message(self, message: str) -> None:
        """동적 UI를 깨뜨리지 않으면서 상태 메시지를 출력한다."""
        if self.dynamic:
            tqdm.write(message, file=self.stream)
        else:
            self.stream.write(message + "\n")
            self.stream.flush()

    def close(self) -> None:
        """tqdm 자원을 한 번만 닫고 마지막 화면 상태를 확정한다."""
        if self.closed:
            return
        self.closed = True
        for line in reversed(self.dashboard_lines):
            line.close()
        if self.progress_bar is not None:
            self.progress_bar.close()


class ExperimentLogger:
    def __init__(
        self,
        output_dir: Path,
        wandb_config: DictConfig,
        resume_state: dict[str, Any] | None = None,
        *,
        console_stream: TextIO | None = None,
        force_tty: bool | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "metrics.jsonl"
        self.config = wandb_config
        self.enabled = bool(wandb_config.enabled)
        self.resume_state = resume_state or {}
        self.run: Any | None = None
        self.last_wandb_step = int(self.resume_state.get("last_logged_step", -1))
        self.console_stream = console_stream if console_stream is not None else sys.stderr
        self.force_tty = force_tty
        self.progress: TrainingProgress | None = None

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
            self.run.define_metric("completed_epoch")
            self.run.define_metric("alignment/*", step_metric="completed_epoch")
            self.run.define_metric("alignment/t2f/*", step_metric="completed_epoch")
            self.run.define_metric("alignment/f2t/*", step_metric="completed_epoch")
            self.run.define_metric("alignment/visualization/*", step_metric="completed_epoch")
            if bool(self.config.watch_model):
                self.run.watch(model, log="gradients", log_freq=int(self.config.log_every))
            self.message(
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

    def start_progress(self, max_steps: int, initial_step: int = 0) -> None:
        """Resume step을 initial로 사용하는 terminal progress UI를 시작한다."""
        self.progress = TrainingProgress(
            max_steps,
            initial_step,
            stream=self.console_stream,
            force_tty=self.force_tty,
        )

    def advance(self, step: int) -> None:
        if self.progress is not None:
            self.progress.advance(step)

    def update_training(self, metrics: dict[str, Any]) -> None:
        if self.progress is not None:
            self.progress.update_training(metrics)

    def update_validation(self, metrics: dict[str, Any]) -> None:
        if self.progress is not None:
            self.progress.update_validation(metrics)

    def update_alignment(self, metrics: dict[str, Any]) -> None:
        if self.progress is not None:
            self.progress.update_alignment(metrics)

    def message(self, message: str) -> None:
        if self.progress is not None:
            self.progress.message(message)
            return
        self.console_stream.write(message + "\n")
        self.console_stream.flush()

    def write(self, record: dict[str, Any]) -> None:
        """전체 record를 JSONL에 보존한다. Terminal 출력은 별도 UI가 담당한다."""
        serialized = json.dumps(record, sort_keys=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")

    def log_wandb(
        self,
        metrics: dict[str, Any],
        step: int,
        *,
        allow_same_step: bool = False,
        images: dict[str, Any] | None = None,
    ) -> None:
        if self.run is None:
            return
        step_decreased = step < self.last_wandb_step
        repeated_without_permission = step == self.last_wandb_step and not allow_same_step
        if step_decreased or repeated_without_permission:
            raise ValueError(
                f"W&B step must increase monotonically: previous={self.last_wandb_step}, next={step}"
            )
        filtered_metrics = filter_wandb_metrics(metrics)
        payload = {"global_step": step, **filtered_metrics}
        if images:
            import wandb

            completed_epoch = metrics.get("completed_epoch")
            for key, image in images.items():
                caption = f"completed_epoch={completed_epoch}, global_step={step}"
                payload[key] = wandb.Image(image, caption=caption)
        self.run.log(payload, step=step)
        self.last_wandb_step = max(self.last_wandb_step, step)

    def wandb_state(self) -> dict[str, Any] | None:
        if self.run is None:
            return None
        return {
            "run_id": self.run.id,
            "run_name": self.run.name,
            "last_logged_step": self.last_wandb_step,
        }

    def close_progress(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress = None

    def finish(self, exit_code: int = 0) -> None:
        self.close_progress()
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None
