"""Shape, gradient, causality, and tiny-overfit checks for TaF-Adapter."""

from __future__ import annotations

import math

import hydra
import torch
from omegaconf import DictConfig

from model.taf_adapter import TaFAdapter
from model.vector_quantizer import VectorQuantizer
from utils.common import validate_config
from utils.config import register_config_schema
from utils.model_factory import build_model


register_config_schema()


def has_finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    return bool(gradients) and all(
        gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
    ) and any(gradient.abs().sum() > 0 for gradient in gradients)


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
    assert (baseline[:, -1] - perturbed[:, -1]).abs().max() > 1e-6, (
        "Summary token did not respond to a past tactile frame"
    )
    print("Causal mask and last-position summary access: PASS")


def check_one_window(
    model: TaFAdapter, window_size: int, device: torch.device, image_size: tuple[int, int],
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
    pressure_indices = outputs["codebook_indices"]["pressure"]
    force_indices = outputs["codebook_indices"]["force"]
    assert int(pressure_indices.min()) >= 0
    assert int(pressure_indices.max()) < model.force_encoder.pressure.quantizer.codebook.num_embeddings
    assert int(force_indices.min()) >= 0
    assert int(force_indices.max()) < model.force_encoder.force.quantizer.codebook.num_embeddings

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


def run_checks(cfg: DictConfig) -> None:
    validate_config(cfg)
    torch.manual_seed(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(cfg.data.image_size)
    pretrained = False if cfg.check.no_pretrained else bool(cfg.tactile_encoder.pretrained)
    print(f"Device: {device}")
    if not pretrained:
        print("NOTE: pretrained weights are disabled; this is a structural model validation")
    check_vq_gradient_routing(device)

    retained_model = None
    target_window = int(cfg.data.window_size)
    window_sizes = [int(value) for value in cfg.check.window_sizes]
    if target_window in window_sizes:
        window_sizes.remove(target_window)
    window_sizes.append(target_window)
    for window_size in window_sizes:
        model = build_model(cfg, window_size=window_size, pretrained=pretrained).to(device)
        expected_beta = float(cfg.vq.commitment_beta)
        assert model.force_encoder.pressure.quantizer.commitment_beta == expected_beta
        assert model.force_encoder.force.quantizer.commitment_beta == expected_beta
        check_one_window(model, window_size, device, image_size)
        if window_size == target_window:
            retained_model = model
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    assert retained_model is not None
    check_causality(retained_model, device)
    encoder = retained_model.tactile_encoder
    print(
        f"Backbone: {encoder.backbone_name}; pretrained={pretrained}; "
        f"features={encoder.backbone_dim}; input={encoder.input_size}; "
        f"mean={encoder.pixel_mean.flatten().tolist()}; std={encoder.pixel_std.flatten().tolist()}"
    )
    if not cfg.check.skip_overfit:
        synthetic_overfit(retained_model, device, int(cfg.check.overfit_steps))
    print("All requested model checks: PASS")


@hydra.main(version_base="1.3", config_path="../config", config_name="taf_adapter")
def main(cfg: DictConfig) -> None:
    run_checks(cfg)


if __name__ == "__main__":
    main()
