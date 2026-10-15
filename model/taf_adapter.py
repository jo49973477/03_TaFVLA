"""TaF-Adapter: tactile-to-force latent alignment (paper Eqs. 1--11)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .force_encoder import ForceEncoder
from .losses import info_nce, retrieval_recall
from .tactile_encoder import TactileEncoder
from .vector_quantizer import VectorQuantizer


class TaFAdapter(nn.Module):
    """Joint tactile encoder and force VQ-VAE used only during pre-training.

    Raw pressure/force inputs are normalized by training-split statistics stored as
    model buffers. At deployment, :meth:`encode_tactile` is the only required path.
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        force_encoder: dict[str, Any] | None = None,
        vq: dict[str, Any] | None = None,
        tactile_encoder: dict[str, Any] | None = None,
        loss: dict[str, Any] | None = None,
        normalization_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        force_cfg = dict(force_encoder or {})
        vq_cfg = dict(vq or {})
        tactile_cfg = dict(tactile_encoder or {})
        loss_cfg = dict(loss or {})
        # vq.commitment_beta is canonical. The fallback preserves direct legacy
        # constructor calls that provided it under loss instead.
        beta = float(vq_cfg.get("commitment_beta", loss_cfg.get("commitment_beta", 0.25)))
        self.force_encoder = ForceEncoder(
            window_size=window_size,
            hidden_dim=int(force_cfg.get("hidden_dim", 512)),
            pressure_latent_dim=int(force_cfg.get("pressure_latent_dim", 128)),
            force_latent_dim=int(force_cfg.get("force_latent_dim", 128)),
            mlp_layers=int(force_cfg.get("mlp_layers", 2)),
            pressure_codebook_size=int(vq_cfg.get("pressure_codebook_size", 512)),
            force_codebook_size=int(vq_cfg.get("force_codebook_size", 512)),
            commitment_beta=beta,
        )
        tactile_cfg.setdefault("output_dim", self.force_encoder.output_dim)
        if int(tactile_cfg["output_dim"]) != self.force_encoder.output_dim:
            raise ValueError("Tactile output_dim must equal pressure_latent_dim + force_latent_dim")
        self.tactile_encoder = TactileEncoder(**tactile_cfg)
        self.window_size = int(window_size)
        self.lambda_quant = float(loss_cfg.get("lambda_quant", 1.0))
        self.lambda_recon = float(loss_cfg.get("lambda_recon", 1.0))
        self.temperature = float(loss_cfg.get("temperature", 0.07))
        self.nce_mode = str(loss_cfg.get("nce_mode", "paper"))
        if self.lambda_quant < 0 or self.lambda_recon < 0:
            raise ValueError("Loss weights must be non-negative")
        if normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive")
        self.normalization_eps = float(normalization_eps)
        self.register_buffer("pressure_mean", torch.zeros(1, 1, 12, 12))
        self.register_buffer("pressure_std", torch.ones(1, 1, 12, 12))
        self.register_buffer("force_mean", torch.zeros(1, 1, 6))
        self.register_buffer("force_std", torch.ones(1, 1, 6))

    @property
    def embedding_dim(self) -> int:
        return self.force_encoder.output_dim

    @torch.no_grad()
    def set_normalization(
        self,
        *,
        pressure_mean: torch.Tensor,
        pressure_std: torch.Tensor,
        force_mean: torch.Tensor,
        force_std: torch.Tensor,
    ) -> None:
        expected = {
            "pressure_mean": (12, 12), "pressure_std": (12, 12),
            "force_mean": (6,), "force_std": (6,),
        }
        values = locals()
        for name, shape in expected.items():
            value = torch.as_tensor(values[name], dtype=getattr(self, name).dtype, device=getattr(self, name).device)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"{name} must be strictly positive")
            getattr(self, name).copy_(value.reshape_as(getattr(self, name)))

    def _normalize(self, pressure: torch.Tensor, force: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pressure_n = (pressure - self.pressure_mean) / self.pressure_std.clamp_min(self.normalization_eps)
        force_n = (force - self.force_mean) / self.force_std.clamp_min(self.normalization_eps)
        return pressure_n, force_n

    def encode_tactile(self, tactile: torch.Tensor) -> torch.Tensor:
        """Inference path: ``(B,T,3,H,W) -> (B,D)`` without force inputs."""
        if tactile.shape[1] != self.window_size:
            raise ValueError(f"Expected T={self.window_size}, got T={tactile.shape[1]}")
        return self.tactile_encoder(tactile)

    def _finish_forward(
        self,
        z_tactile: torch.Tensor,
        force_outputs: dict[str, torch.Tensor],
        pressure_target: torch.Tensor,
        force_target: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        loss_nce, logits, labels = info_nce(
            z_tactile, force_outputs["c_force"], self.temperature, self.nce_mode
        )
        loss_recon_pressure = F.mse_loss(force_outputs["reconstructed_pressure"], pressure_target)
        loss_recon_force = F.mse_loss(force_outputs["reconstructed_force"], force_target)
        loss_recon = loss_recon_pressure + loss_recon_force
        loss_total = loss_nce + self.lambda_quant * force_outputs["loss_quant"] + self.lambda_recon * loss_recon

        normalized_z = F.normalize(z_tactile, dim=-1)
        normalized_c = F.normalize(force_outputs["c_force"], dim=-1)
        similarities = normalized_z @ normalized_c.T
        diagonal = similarities.diagonal()
        if similarities.shape[0] > 1:
            negative = similarities[~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)].mean()
        else:
            negative = similarities.new_zeros(())

        p_quantizer = self.force_encoder.pressure.quantizer
        f_quantizer = self.force_encoder.force.quantizer
        p_used, p_perplexity, _ = VectorQuantizer.usage(
            force_outputs["pressure_indices"], p_quantizer.codebook.num_embeddings
        )
        f_used, f_perplexity, _ = VectorQuantizer.usage(
            force_outputs["force_indices"], f_quantizer.codebook.num_embeddings
        )
        reconstructed_pressure = (
            force_outputs["reconstructed_pressure"] * self.pressure_std + self.pressure_mean
        )
        reconstructed_force = force_outputs["reconstructed_force"] * self.force_std + self.force_mean
        return {
            "z_tactile": z_tactile,
            "c_force": force_outputs["c_force"],
            "c_pressure": force_outputs["c_pressure"],
            "c_force_only": force_outputs["c_force_only"],
            "reconstructed_pressure": reconstructed_pressure,
            "reconstructed_force": reconstructed_force,
            "reconstructed_pressure_normalized": force_outputs["reconstructed_pressure"],
            "reconstructed_force_normalized": force_outputs["reconstructed_force"],
            "loss_nce": loss_nce,
            "loss_quant": force_outputs["loss_quant"],
            "loss_codebook": force_outputs["loss_codebook"],
            "loss_commitment": force_outputs["loss_commitment"],
            "loss_recon_pressure": loss_recon_pressure,
            "loss_recon_force": loss_recon_force,
            "loss_recon": loss_recon,
            "loss_total": loss_total,
            "logits": logits,
            "labels": labels,
            "positive_cosine": diagonal.mean(),
            "negative_cosine": negative,
            "recall_at_1": retrieval_recall(logits, 1),
            "recall_at_5": retrieval_recall(logits, 5),
            "pressure_codebook_usage": p_used.float(),
            "force_codebook_usage": f_used.float(),
            "pressure_codebook_usage_fraction": p_used.float() / p_quantizer.codebook.num_embeddings,
            "force_codebook_usage_fraction": f_used.float() / f_quantizer.codebook.num_embeddings,
            "pressure_codebook_perplexity": p_perplexity,
            "force_codebook_perplexity": f_perplexity,
            "pressure_dead_codes": p_used.new_tensor(p_quantizer.codebook.num_embeddings) - p_used,
            "force_dead_codes": f_used.new_tensor(f_quantizer.codebook.num_embeddings) - f_used,
            "codebook_indices": {
                "pressure": force_outputs["pressure_indices"],
                "force": force_outputs["force_indices"],
            },
        }

    def forward_from_tactile_features(
        self, tactile_features: torch.Tensor, force: torch.Tensor, pressure: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Training path for cached projected ViT features ``(B,T,D_transformer)``."""
        pressure_n, force_n = self._normalize(pressure, force)
        force_outputs = self.force_encoder(pressure_n, force_n)
        z_tactile = self.tactile_encoder.aggregate_features(tactile_features)
        return self._finish_forward(z_tactile, force_outputs, pressure_n, force_n)

    def forward(
        self, tactile: torch.Tensor, force: torch.Tensor, pressure: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        pressure_n, force_n = self._normalize(pressure, force)
        force_outputs = self.force_encoder(pressure_n, force_n)
        z_tactile = self.encode_tactile(tactile)
        return self._finish_forward(z_tactile, force_outputs, pressure_n, force_n)
