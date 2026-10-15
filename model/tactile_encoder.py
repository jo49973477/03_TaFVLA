"""Framewise pretrained ViT plus a causal temporal Transformer and summary token."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TactileEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, transformer_dim: int = 256,
                 transformer_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 backbone: str = "vit_small_patch16_224.augreg_in21k_ft_in1k",
                 pretrained: bool = True, freeze_backbone: bool = True,
                 max_window_size: int = 20):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("TactileEncoder requires timm") from exc
        # Deliberately let pretrained-weight failures propagate: never silently use random weights.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.backbone_dim = int(self.backbone.num_features)
        self.projection = nn.Linear(self.backbone_dim, transformer_dim)
        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        self.input_size = tuple(cfg.get("input_size", (3, 224, 224))[-2:])
        self.interpolation = str(cfg.get("interpolation", "bicubic"))
        self.register_buffer("pixel_mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 1, 3, 1, 1))
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.summary_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        nn.init.trunc_normal_(self.summary_token, std=0.02)
        self.position = nn.Embedding(max_window_size + 1, transformer_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=num_heads, dim_feedforward=4 * transformer_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers, enable_nested_tensor=False)
        self.output = nn.Linear(transformer_dim, output_dim) if transformer_dim != output_dim else nn.Identity()
        self.max_window_size = max_window_size

    @staticmethod
    def causal_mask(window_size: int, device: torch.device | None = None) -> torch.Tensor:
        """True means masked; final summary position can read every image token."""
        length = window_size + 1
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        features = self.encode_frames(tactile)
        return self.aggregate_features(features)

    def encode_frames(self, tactile: torch.Tensor) -> torch.Tensor:
        """Return projected per-frame ViT features with shape ``(B,T,D)``."""
        if tactile.ndim != 5 or tactile.shape[2] != 3:
            raise ValueError(f"Expected tactile (B,T,3,H,W), got {tuple(tactile.shape)}")
        batch, steps = tactile.shape[:2]
        if steps > self.max_window_size:
            raise ValueError(f"T={steps} exceeds max_window_size={self.max_window_size}")
        x = tactile
        if tuple(x.shape[-2:]) != self.input_size:
            x = F.interpolate(x.reshape(batch * steps, 3, *x.shape[-2:]), size=self.input_size,
                              mode=self.interpolation, align_corners=False,
                              antialias=True).reshape(batch, steps, 3, *self.input_size)
        x = (x - self.pixel_mean) / self.pixel_std
        x = x.reshape(batch * steps, 3, *self.input_size)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.projection(features).reshape(batch, steps, -1)

    def aggregate_features(self, features: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        """Causally aggregate ``(B,T,D)`` frame features; the summary token is last."""
        if features.ndim != 3 or features.shape[-1] != self.projection.out_features:
            raise ValueError(
                f"Expected frame features (B,T,{self.projection.out_features}), "
                f"got {tuple(features.shape)}"
            )
        batch, steps = features.shape[:2]
        if steps > self.max_window_size:
            raise ValueError(f"T={steps} exceeds max_window_size={self.max_window_size}")
        summary = self.summary_token.expand(batch, -1, -1)
        tokens = torch.cat((features, summary), dim=1)
        positions = torch.arange(steps + 1, device=tokens.device)
        tokens = tokens + self.position(positions).unsqueeze(0)
        encoded = self.temporal(tokens, mask=self.causal_mask(steps, tokens.device))
        if return_all_tokens:
            return encoded
        return self.output(encoded[:, -1])
