"""Hydra structured schema for the TaF-Adapter experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore


_REGISTERED = False


@dataclass
class DataConfig:
    root_dir: Optional[str] = None
    window_size: int = 5
    stride: int = 1
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    max_sequences: Optional[int] = None
    # Dataset accepts either one sensor name or a list; common validation checks it.
    sensor_types: Any = None
    split_seed: int = 42
    split_unit: str = "sequence"
    validation_fraction: float = 0.1
    skip_invalid_sequences: bool = False
    timestamp_cache_enabled: bool = True
    timestamp_cache_dir: Optional[str] = None
    show_initialization_progress: bool = True
    alignment_decode_windows_per_episode: int = 1
    normalization_eps: float = 1.0e-6
    normalization_chunk_frames: int = 8192


@dataclass
class ForceEncoderConfig:
    hidden_dim: int = 512
    pressure_latent_dim: int = 128
    force_latent_dim: int = 128
    mlp_layers: int = 2


@dataclass
class VQConfig:
    pressure_codebook_size: int = 512
    force_codebook_size: int = 512
    commitment_beta: float = 0.25


@dataclass
class TactileEncoderConfig:
    backbone: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    pretrained: bool = True
    freeze_backbone: bool = True
    transformer_dim: int = 256
    transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_window_size: int = 20


@dataclass
class LossConfig:
    lambda_quant: float = 1.0
    lambda_recon: float = 1.0
    temperature: float = 0.07
    nce_mode: str = "paper"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    max_steps: int = 10000
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    grad_clip_norm: float = 1.0
    log_every: int = 20
    validate_every: int = 500
    validation_batches: Optional[int] = 50
    save_every: int = 1000
    output_dir: str = "outputs/taf_adapter"
    resume: Optional[str] = None
    min_same_episode_window_gap: int = 5


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "TaF-Adapter"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: list[str] = field(default_factory=lambda: ["taf-vla", "pretraining"])
    mode: str = "online"
    log_every: int = 20
    watch_model: bool = False


@dataclass
class AlignmentEvalConfig:
    enabled: bool = True
    retrieval_every_n_epochs: int = 1
    visualize_every_n_epochs: int = 5
    retrieval_max_pairs: int = 256
    visualization_max_pairs: int = 64
    min_window_start_gap: int = 5
    random_seed: int = 42
    run_on_training_end: bool = True
    save_local_figures: bool = True


@dataclass
class CheckConfig:
    overfit_steps: int = 100
    skip_overfit: bool = False
    no_pretrained: bool = False
    window_sizes: list[int] = field(default_factory=lambda: [1, 3, 10, 5])


@dataclass
class TaFConfig:
    data: DataConfig = field(default_factory=DataConfig)
    force_encoder: ForceEncoderConfig = field(default_factory=ForceEncoderConfig)
    vq: VQConfig = field(default_factory=VQConfig)
    tactile_encoder: TactileEncoderConfig = field(default_factory=TactileEncoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    alignment_eval: AlignmentEvalConfig = field(default_factory=AlignmentEvalConfig)
    check: CheckConfig = field(default_factory=CheckConfig)


def register_config_schema() -> None:
    """Register once per process so YAML values are type checked and struct-locked."""
    global _REGISTERED
    if _REGISTERED:
        return
    ConfigStore.instance().store(name="taf_adapter_schema", node=TaFConfig)
    _REGISTERED = True

