"""Shape, gradient, causality, and tiny-overfit checks for TaF-Adapter."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.taf_adapter import TaFAdapter
from model.vector_quantizer import VectorQuantizer


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def build_model(config: dict[str, Any], window_size: int) -> TaFAdapter:
    tactile = dict(config["tactile_encoder"])
    tactile["max_window_size"] = max(int(tactile.get("max_window_size", window_size)), window_size)
    return TaFAdapter(
        window_size=window_size,
        force_encoder=config["force_encoder"],
        vq=config["vq"],
        tactile_encoder=tactile,
        loss=config["loss"],
        normalization_eps=float(config["data"].get("normalization_eps", 1e-6)),
    )


def has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    return bool(gradients) and all(g is not None and torch.isfinite(g).all() for g in gradients) and any(
        g.abs().sum() > 0 for g in gradients
    )


def check_vq_gradient_routing(device: torch.device) -> None:
    quantizer = VectorQuantizer(codebook_size=8, embedding_dim=4, commitment_beta=0.25).to(device)
    z = torch.randn(3, 4, device=device, requires_grad=True)
    result = quantizer(z)
    result["quantized"].square().mean().backward(retain_graph=True)
    assert z.grad is not None and z.grad.abs().sum() > 0, "STE did not route downstream gradient to z"
    assert quantizer.codebook.weight.grad is None, "STE must not update codebook from downstream loss"
    z.grad = None
    result["loss"].backward()
    assert z.grad is not None and z.grad.abs().sum() > 0, "Commitment loss did not update z"
    gradient = quantizer.codebook.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0, "Codebook loss did not update embeddings"
    print("VQ straight-through and stop-gradient routing: PASS")


def check_causality(model: TaFAdapter, device: torch.device) -> None:
    encoder = model.tactile_encoder
    steps = min(5, encoder.max_window_size)
    mask = encoder.causal_mask(steps, device)
    assert mask.shape == (steps + 1, steps + 1)
    assert not mask.diagonal().any()
    assert torch.equal(mask, torch.triu(mask, diagonal=1))
    assert not mask[-1].any(), "Last summary token must attend to all tactile frames"
    assert mask[:-1, -1].all(), "Image tokens must not attend to the future summary token"

    encoder.eval()
    dim = encoder.projection.out_features
    features = torch.randn(2, steps, dim, device=device)
    changed = features.clone()
    changed[:, -1] += 3.0 * torch.randn_like(changed[:, -1])
    with torch.no_grad():
        baseline = encoder.aggregate_features(features, return_all_tokens=True)
        perturbed = encoder.aggregate_features(changed, return_all_tokens=True)
    if steps > 1:
        assert torch.allclose(baseline[:, :-2], perturbed[:, :-2], atol=2e-5, rtol=2e-5), (
            "A future frame changed an earlier token; causal masking is not effective"
        )
    summary_change = (baseline[:, -1] - perturbed[:, -1]).abs().max()
    assert summary_change > 1e-6, "Summary token did not respond to a past tactile frame"
    print("Causal mask and last-position summary access: PASS")


def check_one_window(
    model: TaFAdapter, window_size: int, device: torch.device, image_size: tuple[int, int]
) -> None:
    model.train()
    batch = 2
    tactile = torch.rand(batch, window_size, 3, *image_size, device=device)
    force = torch.randn(batch, window_size, 6, device=device)
    pressure = torch.randn(batch, window_size, 12, 12, device=device)
    outputs = model(tactile=tactile, force=force, pressure=pressure)

    dim = model.embedding_dim
    assert outputs["z_tactile"].shape == (batch, dim)
    assert outputs["c_force"].shape == (batch, dim)
    assert outputs["reconstructed_pressure"].shape == pressure.shape
    assert outputs["reconstructed_force"].shape == force.shape
    for name, value in outputs.items():
        if name.startswith("loss_"):
            assert torch.isfinite(value).all(), f"{name} is not finite"
    p_indices = outputs["codebook_indices"]["pressure"]
    f_indices = outputs["codebook_indices"]["force"]
    assert int(p_indices.min()) >= 0
    assert int(p_indices.max()) < model.force_encoder.pressure.quantizer.codebook.num_embeddings
    assert int(f_indices.min()) >= 0
    assert int(f_indices.max()) < model.force_encoder.force.quantizer.codebook.num_embeddings

    outputs["loss_total"].backward()
    assert has_finite_nonzero_gradient(model.force_encoder.pressure.encoder)
    assert has_finite_nonzero_gradient(model.force_encoder.force.encoder)
    assert has_finite_nonzero_gradient(model.force_encoder.pressure.decoder)
    assert has_finite_nonzero_gradient(model.force_encoder.force.decoder)
    assert has_finite_nonzero_gradient(model.force_encoder.pressure.quantizer.codebook)
    assert has_finite_nonzero_gradient(model.force_encoder.force.quantizer.codebook)
    assert has_finite_nonzero_gradient(model.tactile_encoder.temporal)
    if model.tactile_encoder.freeze_backbone:
        assert all(parameter.grad is None for parameter in model.tactile_encoder.backbone.parameters())
    with torch.no_grad():
        inference = model.encode_tactile(tactile)
    assert inference.shape == (batch, dim)
    print(
        f"T={window_size}: forward/backward PASS; tactile={tuple(outputs['z_tactile'].shape)}, "
        f"force={tuple(outputs['c_force'].shape)}, loss={float(outputs['loss_total'].detach()):.4f}"
    )


def synthetic_overfit(model: TaFAdapter, device: torch.device, steps: int) -> None:
    """Memorize eight fixed pairs while reusing projected frozen-ViT features."""
    torch.manual_seed(123)
    model.train()
    count, window = 8, model.window_size
    feature_dim = model.tactile_encoder.projection.out_features
    tactile_features = torch.randn(count, window, feature_dim, device=device)
    force = torch.randn(count, window, 6, device=device)
    pressure = torch.randn(count, window, 12, 12, device=device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-3,
    )

    model.eval()
    with torch.no_grad():
        initial_output = model.forward_from_tactile_features(tactile_features, force, pressure)
        initial_loss = float(initial_output["loss_total"])
        initial_positive = float(initial_output["positive_cosine"])
    for _ in range(steps):
        model.train()
        output = model.forward_from_tactile_features(tactile_features, force, pressure)
        optimizer.zero_grad(set_to_none=True)
        output["loss_total"].backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_output = model.forward_from_tactile_features(tactile_features, force, pressure)
        final_loss = float(final_output["loss_total"])
        final_positive = float(final_output["positive_cosine"])
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss, f"Synthetic loss did not decrease: {initial_loss} -> {final_loss}"
    assert final_positive > initial_positive, (
        f"Positive cosine did not increase: {initial_positive} -> {final_positive}"
    )
    print(
        f"Synthetic overfit ({count} pairs, {steps} steps): PASS; "
        f"loss {initial_loss:.4f}->{final_loss:.4f}, "
        f"positive cosine {initial_positive:.4f}->{final_positive:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/taf_adapter.yaml"))
    parser.add_argument("--overfit-steps", type=int, default=100)
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="explicitly use random ViT weights for offline structural testing; never automatic",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.no_pretrained:
        config = copy.deepcopy(config)
        config["tactile_encoder"]["pretrained"] = False
        print("NOTE: --no-pretrained explicitly selected; this is not a pretrained-backbone validation")

    torch.manual_seed(int(config["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(config["data"]["image_size"])
    print(f"Device: {device}")
    check_vq_gradient_routing(device)
    retained_model = None
    # Keep T=5 last so its model can be reused by the overfit test.
    for window_size in (1, 3, 10, 5):
        model = build_model(config, window_size).to(device)
        check_one_window(model, window_size, device, image_size)
        if window_size == 5:
            retained_model = model
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    assert retained_model is not None
    check_causality(retained_model, device)
    encoder = retained_model.tactile_encoder
    print(
        f"Backbone: {encoder.backbone_name}; pretrained={config['tactile_encoder']['pretrained']}; "
        f"features={encoder.backbone_dim}; input={encoder.input_size}; "
        f"mean={encoder.pixel_mean.flatten().tolist()}; std={encoder.pixel_std.flatten().tolist()}"
    )
    if not args.skip_overfit:
        synthetic_overfit(retained_model, device, args.overfit_steps)
    print("All requested model checks: PASS")


if __name__ == "__main__":
    main()
