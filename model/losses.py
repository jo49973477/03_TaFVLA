"""Contrastive and reconstruction objectives used by TaF-Adapter."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(tactile: torch.Tensor, force: torch.Tensor, temperature: float = 0.07,
             mode: str = "paper") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mode not in {"paper", "symmetric"}:
        raise ValueError("nce_mode must be 'paper' or 'symmetric'")
    if tactile.ndim != 2 or force.ndim != 2 or tactile.shape != force.shape:
        raise ValueError(
            f"Expected matching (B,D) embeddings, got {tuple(tactile.shape)} and {tuple(force.shape)}"
        )
    z = F.normalize(tactile, dim=-1)
    c = F.normalize(force, dim=-1)
    logits = z @ c.T / temperature
    labels = torch.arange(z.shape[0], device=z.device)
    t2f = F.cross_entropy(logits, labels)
    f2t = F.cross_entropy(logits.T, labels)
    return (t2f if mode == "paper" else 0.5 * (t2f + f2t)), logits, labels


def retrieval_recall(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.shape[1])
    target = torch.arange(logits.shape[0], device=logits.device)
    return (logits.topk(k, dim=1).indices == target[:, None]).any(dim=1).float().mean()
