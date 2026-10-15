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
- `utils/config.py`: Hydra/OmegaConf structured schema. Unknown keys and type
  mismatches fail during config composition.
- `utils/common.py`: seed setup, semantic config validation, and Hydra-safe
  path resolution.
- `utils/model_factory.py`: the single model-construction path shared by
  training and validation.
- `utils/data.py`: leakage-free grouped split, temporal audit, train-only
  normalization, and the unchanged temporal-diverse batch sampler.
- `utils/checkpoint.py`: atomic checkpoint save/load and RNG restoration.
- `utils/logging.py`: the shared console, JSONL, and optional W&B logger.

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

## Hydra configuration

Both training and model validation compose `config/taf_adapter.yaml` through
Hydra and receive an OmegaConf `DictConfig`. The YAML is merged with a small
structured schema, so misspelled keys and incompatible types produce an error.
Semantic checks additionally reject invalid window sizes, learning rates, loss
weights, temperatures, split fractions, and logging intervals.

Inspect the fully composed job config without starting training:

```bash
uv run python -m script.train_adapter --cfg job
```

Any value can be overridden without editing YAML:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  data.window_size=10 \
  training.batch_size=16 \
  training.learning_rate=1e-4 \
  loss.temperature=0.07
```

Hydra is configured with `hydra.job.chdir: false`. Relative dataset,
checkpoint, and training output paths are resolved against the directory from
which the command was launched. Hydra's own run logs go below
`outputs/hydra/`, while model checkpoints and `metrics.jsonl` retain the
independent `training.output_dir` setting (default:
`outputs/taf_adapter`). The VQ commitment coefficient has one source of truth:
`vq.commitment_beta`.

## Validation

Install/synchronize dependencies and run the model tests with:

```bash
uv sync
uv run python -m script.check_model
```

The check covers T=1, 3, 5, and 10; dimensions and reconstructions; finite
losses; backward flow through both force encoders, both codebooks, both
decoders, and the temporal Transformer; absence of frozen-ViT gradients;
codebook index bounds; causal behavior and summary-token access; and a fixed
eight-pair synthetic overfit. The former `--no-pretrained` and
`--skip-overfit` flags are now `check.no_pretrained=true` and
`check.skip_overfit=true`. The overfit duration can be changed with
`check.overfit_steps=20`.

Fast refactor/config/checkpoint/logger tests run with:

```bash
uv run python -m unittest discover -s tests -v
```

## Dataset gate and training

Before training, inspect a downloaded dataset directly:

```bash
uv run python -m script.check_dataset \
  --root-dir /path/to/TaF-Dataset \
  --test-window-sizes
```

Then start pre-training:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset
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
config, step/epoch, and RNG state. Resume with:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  training.resume=outputs/taf_adapter/last.pt
```

Metrics are printed and appended to
`outputs/taf_adapter/metrics.jsonl` by default.

## Weights & Biases

W&B is disabled by default and therefore requires neither authentication nor a
network connection. Authenticate separately with `wandb login` or the
`WANDB_API_KEY` environment variable, then opt in:

```bash
uv run wandb login
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  training.max_steps=1000 \
  wandb.enabled=true
```

For a network-free local W&B run:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  wandb.enabled=true \
  wandb.mode=offline
```

The final resolved Hydra config and the W&B run ID/name are recorded. Training
and validation log total, InfoNCE, quantization, codebook, commitment, pressure
and force reconstruction losses; positive/negative cosine; recall@1/5; both
codebooks' used count, usage fraction, perplexity, and dead-code count. W&B
also receives learning rate, gradient norm, epoch, elapsed time, and the actual
optimizer step as its global step. No additional model forward pass is made for
logging, and datasets/checkpoints are not uploaded as artifacts.

`training.log_every` controls console and JSONL aggregation,
`training.validate_every` controls validation, and `wandb.log_every` controls
training metric uploads to W&B. Validation uses its current optimizer step and
is merged into the same W&B log call when the intervals coincide. A checkpoint
stores the W&B run ID; resume continues that run when present, while older
checkpoints without the field start a new run. Online initialization errors are
reported rather than silently changing mode. Normal and exceptional shutdowns
call `wandb.finish()` after successful initialization.

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
