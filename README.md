# TaF-Adapter pre-training

This repository implements only the tactile-force alignment adapter from
[TaF-VLA](https://arxiv.org/abs/2601.20321), Section IV-A and Eqs. (1)--(11).
It does not implement the VLA policy, action prediction, or downstream robot
training.

## Architecture

- `model/force_encoder.py`: independent pressure and 6-axis wrench MLP
  encoder/VQ/decoder branches. A complete temporal window is flattened before
  each MLP, and the two quantized vectors are concatenated.
- `model/vector_quantizer.py`: Euclidean nearest-neighbor lookup, the standard
  VQ-VAE codebook and commitment losses, and a straight-through estimator.
- `model/tactile_encoder.py`: framewise pretrained ViT-Small followed by a
  causal temporal Transformer. The learnable summary token is last, so its
  causal-attention row can attend to all tactile frames.
- `model/taf_adapter.py`: normalization, both encoders, Eq. (11), metrics, and
  the tactile-only `encode_tactile()` inference API.
- `model/losses.py`: cosine InfoNCE and retrieval recall.

For the default `T=5`, the main shapes are:

```text
pressure (B,5,12,12) -> z_pressure (B,128) -> VQ -> reconstruction (B,5,12,12)
force       (B,5,6)  -> z_force    (B,128) -> VQ -> reconstruction (B,5,6)
concat(two VQ vectors)                                      -> c_force (B,256)
tactile (B,5,3,H,W) -> ViT frames (B,5,384) -> projection/causal TF -> z_tactile (B,256)
```

The default backbone is
`vit_small_patch16_224.augreg_in21k_ft_in1k`, verified against the installed
`timm==1.0.30` pretrained registry. It uses AugReg ImageNet-21k pretraining and
ImageNet-1k fine-tuning, a 384-dimensional output, 224x224 inputs, and mean/std
of 0.5 per RGB channel. Pretrained-load errors are propagated; the code never
silently falls back to random initialization. The backbone is frozen by
default, while projection and temporal Transformer layers remain trainable.

## Objectives and gradient flow

The forward value of each straight-through VQ vector is the nearest codebook
entry. Its backward derivative is routed to the force encoder. Therefore,
reconstruction and InfoNCE gradients train the force encoders; reconstruction
also trains each decoder; InfoNCE trains the tactile projection/Transformer;
and the explicit codebook term trains each independent codebook. The
commitment term trains the force encoders. A frozen ViT receives no gradients.

`loss_total = loss_nce + lambda_quant * loss_quant + lambda_recon * loss_recon`.
The default weights are all 1.0 and commitment beta is 0.25. These are initial
experimental choices, not values reported by the paper, and are kept explicit
so loss scales can be inspected before changing them. Reconstruction uses an
element-wise mean MSE independently for normalized pressure and force, another
documented implementation choice because the paper does not publish its
reduction convention.

The paper calls InfoNCE symmetric in prose, but Eq. (10) contains only the
tactile-to-force direction. `nce_mode: paper` implements that equation;
`nce_mode: symmetric` averages tactile-to-force and force-to-tactile losses.

## Validation

Install/synchronize dependencies and run the model tests with:

```bash
uv sync
uv run script/check_model.py
```

The check covers T=1, 3, 5, and 10; dimensions and reconstructions; finite
losses; backward flow through both force encoders, both codebooks, both
decoders, and the temporal Transformer; absence of frozen-ViT gradients;
codebook index bounds; causal behavior and summary-token access; and a fixed
eight-pair synthetic overfit. `--no-pretrained` is an explicit offline-only
structural test option and is never selected automatically.

## Dataset gate and training

Before training, inspect a downloaded dataset directly:

```bash
uv run script/check_dataset.py --root-dir /path/to/TaF-Dataset --test-window-sizes
```

Then start pre-training:

```bash
uv run script/train_adapter.py \
  --config config/taf_adapter.yaml \
  --root-dir /path/to/TaF-Dataset
```

`TaFDataset` validates episode index ranges and every Parquet-to-MP4 timestamp
match during construction. The trainer additionally decodes representative
windows from every episode and refuses to proceed if an invalid sequence was
skipped. Train/validation splitting happens at sequence or episode granularity,
never at window granularity.

Pressure mean/std is stored per 12x12 taxel and wrench mean/std per ordered 6D
component. Statistics use unique frames from the training groups only, not
overlapping windows and never validation data. They are stored both in model
buffers and checkpoint metadata. A batch sampler greedily separates nearby
windows from the same episode when alternatives exist. This reduces adjacent
false negatives, but cannot eliminate semantically similar negatives across a
continuous dataset. Gradient accumulation also does not enlarge the InfoNCE
negative set: only pairs in one forward pass are negatives.

Checkpoints contain model/optimizer/scaler state, normalization, split groups,
config, step/epoch, and RNG state. Resume with `--resume PATH`. Metrics are
printed and appended to `outputs/taf_adapter/metrics.jsonl` by default.

## Reproduction boundary

Directly reproduced from the paper: two independent force modalities and
codebooks, nearest-neighbor VQ, concatenation, VQ-VAE loss, reconstruction,
framewise ViT features, temporal causal Transformer with summary token, cosine
InfoNCE, Eq. (11), and the reported `T=5`.

Project choices because the paper does not specify them: exact ViT checkpoint
and freeze policy, MLP depth/width, latent/codebook sizes, Transformer details,
normalization and reconstruction reduction, optimizer/schedule settings, loss
weights/temperature, and whether to use the prose-symmetric rather than
equation-directional InfoNCE. All are visible in `config/taf_adapter.yaml`.
