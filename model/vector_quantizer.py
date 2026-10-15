"""Nearest-neighbor vector quantization with the VQ-VAE straight-through path."""

from __future__ import annotations

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, commitment_beta: float = 0.25):
        super().__init__()
        if codebook_size < 1 or embedding_dim < 1:
            raise ValueError("codebook_size and embedding_dim must be positive")
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.commitment_beta = commitment_beta

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.shape[-1] != self.codebook.embedding_dim:
            raise ValueError(
                f"Expected latent dimension {self.codebook.embedding_dim}, got {z.shape[-1]}"
            )
        shape = z.shape
        flat = z.reshape(-1, shape[-1])
        weight = self.codebook.weight
        distance = (flat.square().sum(1, keepdim=True)
                    + weight.square().sum(1).unsqueeze(0)
                    - 2 * flat @ weight.t())
        indices = distance.argmin(dim=1)
        quantized = self.codebook(indices).view(shape)
        codebook_loss = (quantized - z.detach()).square().sum(dim=-1).mean()
        commitment_loss = (z - quantized.detach()).square().sum(dim=-1).mean()
        loss = codebook_loss + self.commitment_beta * commitment_loss
        straight_through = z + (quantized - z).detach()
        return {
            "quantized": straight_through,
            "codebook_vectors": quantized,
            "indices": indices.view(shape[:-1]),
            "loss": loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

    @staticmethod
    def usage(indices: torch.Tensor, codebook_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = torch.bincount(indices.reshape(-1), minlength=codebook_size).float()
        probs = counts / counts.sum().clamp_min(1)
        used = (counts > 0).sum()
        perplexity = torch.exp(-(probs * probs.clamp_min(1e-10).log()).sum())
        return used, perplexity, counts
