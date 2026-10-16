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

### Persistent video timestamp cache

Dataset construction still obtains every presentation timestamp with the exact
`ffprobe -show_frames` strategy used for temporal alignment; timestamps are
never estimated from FPS or frame number. The first request for each
metadata-referenced MP4 stores the ordered `float64` result under
`${XDG_CACHE_HOME:-~/.cache}/taf-vla/video_timestamps/`. A later run validates
the canonical absolute path, file size, nanosecond modification time, cache
format version, extraction-strategy version, array shape/dtype/finiteness,
strict ordering, and an array checksum before accepting a hit.

Each video is committed immediately as one `.npz` via a temporary file and
atomic rename. A per-video file lock makes concurrent initializers re-check the
finished cache instead of reading partial data. Consequently, Ctrl+C preserves
all videos completed before the interrupted probe. Stale or corrupt entries are
regenerated independently and do not require clearing the whole cache. The
source dataset is never modified.

The initialization UI counts only MP4s referenced by the selected LeRobot
metadata. On an interactive TTY it uses fixed-position tqdm rows for sequences,
videos, elapsed time/ETA, and cache hit/miss/failure counts. Non-TTY jobs emit
sparse plain-text summaries without ANSI cursor control. Configure this through
Hydra:

```yaml
data:
  timestamp_cache_enabled: true
  timestamp_cache_dir: null
  show_initialization_progress: true
```

`script.check_dataset` reuses the same default cache. Its corresponding flags
are `--[no-]timestamp-cache-enabled`, `--timestamp-cache-dir`, and
`--[no-]show-initialization-progress`. Disabling the cache runs the same
ffprobe extraction every time.

Some TaF-Dataset distributions omit `meta/episodes` for otherwise complete
sequences. The loader uses official episode metadata whenever it exists. If it
is missing, a deliberately narrow fallback supports only one data Parquet and
one `observation.image` MP4. It reconstructs boundaries in memory only after
checking contiguous global indices, episode IDs, per-episode frame indices,
timestamps, FPS, `total_frames`, `total_episodes`, and `info.json` path
templates. The normal ffprobe timestamp tolerance and one-to-one frame mapping
checks still run afterward. A warning identifies every reconstructed sequence;
no file is added to or changed in the source dataset.

Pressure mean/std is stored per 12x12 taxel and wrench mean/std per ordered 6D
component. Statistics use unique frames from the training groups only, not
overlapping windows and never validation data. They are stored both in model
buffers and checkpoint metadata. A batch sampler greedily separates nearby
windows from the same episode when alternatives exist. This reduces adjacent
false negatives, but cannot eliminate semantically similar negatives across a
continuous dataset. Gradient accumulation also does not enlarge the InfoNCE
negative set: only pairs in one forward pass are negatives.

Checkpoints contain model/optimizer/scaler state, normalization, split groups,
config, step/epoch, alignment-evaluation state, and RNG state. Older checkpoints
without alignment metadata remain loadable. Resume with:

```bash
uv run python -m script.train_adapter \
  data.root_dir=/path/to/TaF-Dataset \
  training.resume=outputs/taf_adapter/last.pt
```

Metrics are appended to `outputs/taf_adapter/metrics.jsonl` by default. An
interactive terminal shows optimizer progress and a fixed training/validation
dashboard instead of printing each JSON dictionary.

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
and force reconstruction losses; positive/negative cosine; recall@1/5; and
pressure/force codebook perplexity. Detailed codebook used count, usage
fraction, and dead-code count are intentionally excluded from W&B, but the
model still computes them and the unfiltered values remain in `metrics.jsonl`.
W&B also receives learning rate, gradient norm, epoch, elapsed time, and the
actual optimizer step as its global step. No additional model forward pass is
made for logging, and datasets/checkpoints are not uploaded as artifacts.

W&B's Python SDK can define a common step metric, but does not reliably impose
a particular multi-run workspace panel layout. The logger therefore uploads
each native scalar once and never re-uploads the complete history as a custom
chart. In a run workspace, use **Add panel → Line plot**, choose
`global_step` for X, and select these Y series:

- **Training Loss Components**: `train/loss_total`, `train/loss_nce`,
  `train/loss_quant`, `train/loss_recon`
- **Validation Loss Components**: `val/loss_total`, `val/loss_nce`,
  `val/loss_quant`, `val/loss_recon`
- **Training Codebook Perplexity**: `train/pressure_codebook_perplexity`,
  `train/force_codebook_perplexity`
- **Validation Codebook Perplexity**: `val/pressure_codebook_perplexity`,
  `val/force_codebook_perplexity`
- **Cross-Modal Retrieval**: `alignment/t2f/recall_at_1`,
  `alignment/t2f/recall_at_5`, `alignment/t2f/mrr`,
  `alignment/t2f/hard_negative_margin` with `completed_epoch` as X

Alignment images are logged under `alignment/visualization/*` with both
`completed_epoch` and the actual optimizer `global_step` in their record and
caption.

Each panel overlays the listed series on one chart. The `define_metric` calls
bind both `train/*` and `val/*` histories to the same monotonic `global_step`.

`training.log_every` controls terminal dashboard and JSONL aggregation,
`training.validate_every` controls validation, and `wandb.log_every` controls
training metric uploads to W&B. Validation uses its current optimizer step and
is merged into the same W&B log call when the intervals coincide. A checkpoint
stores the W&B run ID; resume continues that run when present, while older
checkpoints without the field start a new run. Online initialization errors are
reported rather than silently changing mode. Normal and exceptional shutdowns
call `wandb.finish()` after successful initialization.

On a TTY, tqdm reserves one progress row and eight fixed metric rows for loss,
validation, and cross-modal retrieval. Progress
counts completed optimizer updates, so gradient-accumulation micro-steps do not
advance it; resume starts at the checkpoint step. Elapsed time, ETA, completion
percentage, and steps/second are shown by the progress row. In non-TTY contexts
(redirected output, CI, or a terminal multiplexer without usable TTY support),
ANSI cursor movement is disabled and a plain-text summary is emitted only at
the configured logging/validation intervals.

## Cross-modal alignment evaluation

Training uses mini-batch InfoNCE diagnostics, while the alignment evaluator uses
a separate, fixed validation candidate set. It extracts the model's
`z_tactile` and `c_force` under `model.eval()` and `torch.inference_mode()`, L2
normalizes both, and computes the complete cosine matrix
`S = z_tactile @ c_force.T`. Metrics are reported for both tactile→force and
force→tactile:

- Recall@1 and Recall@5 over the complete fixed candidate set
- mean reciprocal rank (MRR)
- mean matched-pair and non-matched-pair cosine similarity
- mean matched similarity minus the hardest negative similarity

Only validation indices produced by the existing group split are eligible.
Within one episode, selected window starts are separated by at least
`max(alignment_eval.min_window_start_gap, data.window_size)`. Episode queues are
sampled round-robin with a local fixed seed, and the exact global indices,
episode identifiers, starts, effective gap, and seed are saved to
`outputs/taf_adapter/alignment/validation_selection.json`. This removes direct
frame overlap, but temporally distant windows can still represent the same
physical force state. Retrieval scores must therefore not be interpreted as a
perfect semantic-duplicate-free benchmark.

The default configuration is:

```yaml
alignment_eval:
  enabled: true
  retrieval_every_n_epochs: 1
  visualize_every_n_epochs: 5
  retrieval_max_pairs: 256
  visualization_max_pairs: 64
  min_window_start_gap: 5
  random_seed: 42
  run_on_training_end: true
  save_local_figures: true
```

Options are regular Hydra overrides:

```fish
uv run python -m script.train_adapter \
    data.root_dir=/path/to/TaF-Dataset \
    alignment_eval.visualize_every_n_epochs=10

uv run python -m script.train_adapter \
    data.root_dir=/path/to/TaF-Dataset \
    alignment_eval.enabled=false
```

An epoch metric is emitted only after every training batch for that epoch was
consumed. `completed_epoch` is 1-based and is stored in checkpoints. If
`max_steps` stops inside an epoch, `run_on_training_end=true` performs a
retrieval-only record labeled `final`; if no epoch completed, the terminal UI
explicitly shows `Final (no completed epoch)`. An evaluation already performed
at the same final model step is not duplicated.

Every visualization interval produces:

- `epoch_XXXX_similarity.png`: tactile rows × force/pressure columns, fixed
  cosine range `[-1, 1]`; matched pairs lie on the diagonal.
- `epoch_XXXX_umap.png`: one joint UMAP fit over both modalities, with matching
  indices connected. Each epoch refits UMAP, so coordinates across epochs are
  not a quantitative distance measure.
- `epoch_XXXX_distribution.png`: density-normalized positive, negative, and
  hard-negative cosine distributions with positive/negative means.

PNG files are saved under `outputs/taf_adapter/alignment/` and the same rendered
images are uploaded once at their evaluation step when W&B is enabled. UMAP is
CPU-side and requires `umap-learn`; install a missing dependency with
`uv add umap-learn`. Runtime is dominated by encoding at most
`retrieval_max_pairs` windows, an `N×N` cosine matrix, and periodic UMAP over at
most twice `visualization_max_pairs` points. Embeddings are moved to CPU and all
Matplotlib figures are closed after rendering.

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
