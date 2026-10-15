"""Separate VQ-VAE branches for pressure maps and six-axis force/torque."""

from __future__ import annotations

import torch
from torch import nn

from .vector_quantizer import VectorQuantizer


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    if layers < 1:
        raise ValueError("mlp_layers must be >= 1")
    modules: list[nn.Module] = []
    current = in_dim
    for _ in range(layers - 1):
        modules.extend((nn.Linear(current, hidden_dim), nn.GELU()))
        current = hidden_dim
    modules.append(nn.Linear(current, out_dim))
    return nn.Sequential(*modules)


class _QuantizedBranch(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, layers: int,
                 codebook_size: int, beta: float):
        super().__init__()
        self.encoder = _mlp(input_dim, latent_dim, hidden_dim, layers)
        self.quantizer = VectorQuantizer(codebook_size, latent_dim, beta)
        self.decoder = _mlp(latent_dim, input_dim, hidden_dim, layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x.flatten(start_dim=1))
        quant = self.quantizer(z)
        reconstruction = self.decoder(quant["quantized"]).view_as(x)
        return {**quant, "z": z, "reconstruction": reconstruction}


class ForceEncoder(nn.Module):
    """Encode (B,T,12,12) and (B,T,6), with independent VQ codebooks."""

    def __init__(self, window_size: int, hidden_dim: int = 512,
                 pressure_latent_dim: int = 128, force_latent_dim: int = 128,
                 mlp_layers: int = 2, pressure_codebook_size: int = 512,
                 force_codebook_size: int = 512, commitment_beta: float = 0.25):
        super().__init__()
        self.window_size = window_size
        self.pressure = _QuantizedBranch(window_size * 144, pressure_latent_dim, hidden_dim,
                                         mlp_layers, pressure_codebook_size, commitment_beta)
        self.force = _QuantizedBranch(window_size * 6, force_latent_dim, hidden_dim,
                                      mlp_layers, force_codebook_size, commitment_beta)
        self.output_dim = pressure_latent_dim + force_latent_dim

    def forward(self, pressure: torch.Tensor, force: torch.Tensor) -> dict[str, torch.Tensor]:
        if pressure.ndim != 4 or pressure.shape[1:] != (self.window_size, 12, 12):
            raise ValueError(f"Expected pressure (B,{self.window_size},12,12), got {tuple(pressure.shape)}")
        if force.ndim != 3 or force.shape[1:] != (self.window_size, 6):
            raise ValueError(f"Expected force (B,{self.window_size},6), got {tuple(force.shape)}")
        p = self.pressure(pressure)
        f = self.force(force)
        return {
            "z_pressure": p["z"], "z_force": f["z"],
            "c_pressure": p["quantized"], "c_force_only": f["quantized"],
            "c_force": torch.cat((p["quantized"], f["quantized"]), dim=-1),
            "reconstructed_pressure": p["reconstruction"],
            "reconstructed_force": f["reconstruction"],
            "loss_quant": p["loss"] + f["loss"],
            "loss_quant_pressure": p["loss"], "loss_quant_force": f["loss"],
            "loss_codebook": p["codebook_loss"] + f["codebook_loss"],
            "loss_codebook_pressure": p["codebook_loss"],
            "loss_codebook_force": f["codebook_loss"],
            "loss_commitment": p["commitment_loss"] + f["commitment_loss"],
            "loss_commitment_pressure": p["commitment_loss"],
            "loss_commitment_force": f["commitment_loss"],
            "pressure_indices": p["indices"], "force_indices": f["indices"],
        }
