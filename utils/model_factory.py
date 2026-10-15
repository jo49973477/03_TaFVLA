"""Shared TaF-Adapter construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from model.taf_adapter import TaFAdapter


def _plain_mapping(value: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        output = OmegaConf.to_container(value, resolve=True)
    else:
        output = dict(value)
    if not isinstance(output, dict):
        raise TypeError("Model config section must be a mapping")
    return output


def build_model(
    config: DictConfig | Mapping[str, Any], *, window_size: int | None = None,
    pretrained: bool | None = None,
) -> TaFAdapter:
    """Build the unchanged model while centralizing config adaptation."""
    data_cfg = _plain_mapping(config["data"])
    force_cfg = _plain_mapping(config["force_encoder"])
    vq_cfg = _plain_mapping(config["vq"])
    tactile_cfg = _plain_mapping(config["tactile_encoder"])
    loss_cfg = _plain_mapping(config["loss"])
    selected_window = int(data_cfg["window_size"] if window_size is None else window_size)
    tactile_cfg["max_window_size"] = max(
        int(tactile_cfg.get("max_window_size", selected_window)), selected_window,
    )
    if pretrained is not None:
        tactile_cfg["pretrained"] = pretrained

    return TaFAdapter(
        window_size=selected_window,
        force_encoder=force_cfg,
        vq=vq_cfg,
        tactile_encoder=tactile_cfg,
        loss=loss_cfg,
        normalization_eps=float(data_cfg.get("normalization_eps", 1e-6)),
    )

