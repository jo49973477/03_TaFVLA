"""Cross-modal retrieval evaluation and visualization for TaF-Adapter."""

from __future__ import annotations

import json
import random
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class AlignmentSelection:
    """한 번 선택한 고정 validation retrieval 후보와 재현성 metadata."""

    indices: list[int]
    samples: list[dict[str, Any]]
    random_seed: int
    requested_max_pairs: int
    requested_min_gap: int
    effective_min_gap: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "random_seed": self.random_seed,
            "requested_max_pairs": self.requested_max_pairs,
            "selected_pairs": len(self.indices),
            "requested_min_window_start_gap": self.requested_min_gap,
            "effective_min_window_start_gap": self.effective_min_gap,
            "indices": self.indices,
            "samples": self.samples,
        }


@dataclass
class AlignmentResult:
    """CPU에 보관하는 정규화 embedding, similarity matrix, retrieval metric."""

    tactile: torch.Tensor
    force: torch.Tensor
    similarity: torch.Tensor
    metrics: dict[str, float]


def select_alignment_indices(
    dataset: Any,
    validation_indices: list[int],
    *,
    max_pairs: int,
    min_window_start_gap: int,
    random_seed: int,
    training_indices: list[int] | None = None,
) -> AlignmentSelection:
    """Validation window만 사용해 episode-balanced 고정 후보를 선택한다.

    같은 episode에서는 window 시작점 차이가 적어도 window 길이 이상이 되게
    먼저 거른다. 이후 episode queue를 round-robin으로 순회해 긴 episode 한
    곳이 retrieval set을 독점하지 않도록 한다. 전역 RNG는 사용하지 않는다.
    """
    if max_pairs < 2:
        raise ValueError("alignment retrieval requires max_pairs >= 2")
    if min_window_start_gap < 1:
        raise ValueError("min_window_start_gap must be >= 1")
    validation_set = set(validation_indices)
    if training_indices is not None and validation_set.intersection(training_indices):
        raise ValueError("Training indices leaked into the validation retrieval candidates")

    window_size = int(dataset.window_size)
    effective_gap = max(min_window_start_gap, window_size)
    episode_ids = {id(episode): index for index, episode in enumerate(dataset.episodes)}
    grouped: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    for global_index in sorted(validation_set):
        episode, start = dataset._locate(global_index)
        key = (str(episode.sequence_id), int(episode.episode_index), episode_ids[id(episode)])
        grouped[key].append((global_index, int(start)))

    rng = random.Random(random_seed)
    by_sequence: dict[str, list[deque[tuple[int, int, tuple[str, int, int]]]]] = defaultdict(list)
    for key, items in sorted(grouped.items()):
        spaced: list[tuple[int, int]] = []
        for item in sorted(items, key=lambda value: value[1]):
            if not spaced or item[1] - spaced[-1][1] >= effective_gap:
                spaced.append(item)
        rng.shuffle(spaced)
        by_sequence[key[0]].append(deque((index, start, key) for index, start in spaced))

    sequence_names = sorted(by_sequence)
    rng.shuffle(sequence_names)
    episode_queues: list[deque[tuple[int, int, tuple[str, int, int]]]] = []
    remaining = True
    round_index = 0
    while remaining:
        remaining = False
        for sequence_name in sequence_names:
            episodes = by_sequence[sequence_name]
            if round_index < len(episodes):
                episode_queues.append(episodes[round_index])
                remaining = True
        round_index += 1

    selected: list[tuple[int, int, tuple[str, int, int]]] = []
    queues = deque(episode_queues)
    while queues and len(selected) < max_pairs:
        queue = queues.popleft()
        if queue:
            selected.append(queue.popleft())
        if queue:
            queues.append(queue)

    if len(selected) < max_pairs:
        warnings.warn(
            f"Alignment retrieval requested {max_pairs} pairs, but only "
            f"{len(selected)} non-overlapping validation pairs are available "
            f"with effective start gap {effective_gap}.",
            RuntimeWarning,
        )
    if len(selected) < 2:
        raise RuntimeError("Alignment retrieval needs at least two validation pairs")

    samples = [
        {
            "dataset_index": index,
            "sequence_id": key[0],
            "episode_index": key[1],
            "episode_list_index": key[2],
            "window_start": start,
        }
        for index, start, key in selected
    ]
    return AlignmentSelection(
        indices=[item[0] for item in selected],
        samples=samples,
        random_seed=random_seed,
        requested_max_pairs=max_pairs,
        requested_min_gap=min_window_start_gap,
        effective_min_gap=effective_gap,
    )


def save_alignment_selection(selection: AlignmentSelection, path: Path) -> None:
    """고정 후보의 index, episode, start, seed를 JSON으로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection.as_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _direction_metrics(similarity: torch.Tensor) -> dict[str, float]:
    """한 retrieval 방향의 Recall@K, MRR, cosine separation을 계산한다."""
    count = similarity.shape[0]
    if similarity.ndim != 2 or similarity.shape[1] != count or count < 2:
        raise ValueError("similarity must be a square matrix with at least two pairs")
    target = torch.arange(count)
    ranking = torch.argsort(similarity, dim=1, descending=True, stable=True)
    ranks = torch.argmax((ranking == target[:, None]).to(torch.int64), dim=1) + 1
    positive = similarity.diagonal()
    negative_mask = ~torch.eye(count, dtype=torch.bool)
    negatives = similarity[negative_mask]
    hardest_negative = similarity.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
    return {
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= min(5, count)).float().mean()),
        "mrr": float(ranks.float().reciprocal().mean()),
        "positive_cosine": float(positive.mean()),
        "negative_cosine": float(negatives.mean()),
        "hard_negative_margin": float((positive - hardest_negative).mean()),
    }


def compute_alignment_metrics(
    tactile_embeddings: torch.Tensor,
    force_embeddings: torch.Tensor,
) -> AlignmentResult:
    """L2 정규화 후 양방향 전체-candidate retrieval metric을 계산한다."""
    if tactile_embeddings.ndim != 2 or force_embeddings.ndim != 2:
        raise ValueError("Alignment embeddings must both have shape (N, D)")
    if tactile_embeddings.shape != force_embeddings.shape:
        raise ValueError(
            f"Embedding shapes differ: {tuple(tactile_embeddings.shape)} vs "
            f"{tuple(force_embeddings.shape)}"
        )
    tactile = F.normalize(tactile_embeddings.float().cpu(), p=2, dim=1)
    force = F.normalize(force_embeddings.float().cpu(), p=2, dim=1)
    similarity = tactile @ force.T
    metrics = {
        f"alignment/t2f/{name}": value
        for name, value in _direction_metrics(similarity).items()
    }
    metrics.update({
        f"alignment/f2t/{name}": value
        for name, value in _direction_metrics(similarity.T).items()
    })
    return AlignmentResult(tactile, force, similarity, metrics)


def collect_alignment_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """eval/inference_mode에서 고정 loader의 z_tactile과 c_force를 수집한다."""
    was_training = model.training
    tactile_embeddings: list[torch.Tensor] = []
    force_embeddings: list[torch.Tensor] = []
    try:
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                tactile = batch["tactile"].to(device, non_blocking=True)
                force = batch["force"].to(device, non_blocking=True)
                pressure = batch["pressure"].to(device, non_blocking=True)
                outputs = model(tactile=tactile, force=force, pressure=pressure)
                tactile_embeddings.append(outputs["z_tactile"].detach().float().cpu())
                force_embeddings.append(outputs["c_force"].detach().float().cpu())
    finally:
        model.train(was_training)
    if not tactile_embeddings:
        raise RuntimeError("Alignment evaluation loader produced no batches")
    return torch.cat(tactile_embeddings), torch.cat(force_embeddings)


def evaluate_alignment(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> AlignmentResult:
    """고정 validation subset의 embedding을 추출하고 retrieval을 평가한다."""
    tactile, force = collect_alignment_embeddings(model, loader, device)
    return compute_alignment_metrics(tactile, force)


def similarity_distributions(
    similarity: torch.Tensor,
    *,
    max_negatives: int = 100_000,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Positive, all-pair negative, row-wise hard-negative cosine을 반환한다."""
    count = similarity.shape[0]
    if similarity.ndim != 2 or similarity.shape != (count, count) or count < 2:
        raise ValueError("similarity must be square with at least two pairs")
    values = similarity.detach().float().cpu()
    diagonal = torch.eye(count, dtype=torch.bool)
    positive = values.diagonal().numpy().copy()
    negative = values[~diagonal].numpy().copy()
    hard_negative = values.masked_fill(diagonal, -torch.inf).max(dim=1).values.numpy().copy()
    if len(negative) > max_negatives:
        rng = np.random.default_rng(random_seed)
        chosen = rng.choice(len(negative), size=max_negatives, replace=False)
        negative = negative[chosen]
    return {
        "positive": positive,
        "negative": negative,
        "hard_negative": hard_negative,
    }


def joint_umap_projection(
    tactile: torch.Tensor,
    force: torch.Tensor,
    *,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """두 modality를 합친 하나의 UMAP fit으로 2D 좌표를 계산한다."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "Joint UMAP visualization requires umap-learn. "
            "Install it with `uv add umap-learn`."
        ) from exc
    if tactile.shape != force.shape or tactile.ndim != 2:
        raise ValueError("UMAP inputs must have matching shape (N, D)")
    count = tactile.shape[0]
    combined = torch.cat((tactile, force), dim=0).float().cpu().numpy()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(15, len(combined) - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=random_seed,
        n_jobs=1,
    )
    coordinates = np.asarray(reducer.fit_transform(combined), dtype=np.float32)
    return coordinates[:count], coordinates[count:]


def matching_segments(tactile_2d: np.ndarray, force_2d: np.ndarray) -> np.ndarray:
    """동일 index의 UMAP matching pair를 잇는 (N, 2 endpoints, 2D) 선분."""
    if tactile_2d.shape != force_2d.shape or tactile_2d.ndim != 2:
        raise ValueError("Matching UMAP coordinates must have equal shape (N, 2)")
    if tactile_2d.shape[1] != 2:
        raise ValueError("UMAP coordinates must be two-dimensional")
    return np.stack((tactile_2d, force_2d), axis=1)


def _figure_image(fig: Any, path: Path | None) -> np.ndarray:
    """선택적으로 PNG를 저장하고 W&B용 RGB 배열을 만든 뒤 Figure를 닫는다."""
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def create_alignment_visualizations(
    result: AlignmentResult,
    *,
    max_pairs: int,
    random_seed: int,
    title_label: str,
    file_prefix: str,
    output_dir: Path,
    save_local_figures: bool,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Similarity matrix, joint UMAP, similarity distribution PNG를 생성한다."""
    count = min(max_pairs, len(result.tactile))
    if count < 2:
        raise ValueError("Visualization requires at least two alignment pairs")
    tactile = result.tactile[:count]
    force = result.force[:count]
    similarity = tactile @ force.T
    paths: dict[str, str] = {}

    def path_for(suffix: str) -> Path | None:
        if not save_local_figures:
            return None
        path = output_dir / f"{file_prefix}_{suffix}.png"
        paths[suffix] = str(path)
        return path

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(similarity.numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xlabel("Force/Pressure Embedding Index")
    ax.set_ylabel("Tactile Embedding Index")
    recall = result.metrics["alignment/t2f/recall_at_1"]
    ax.set_title(f"Cosine Similarity Matrix — {title_label} — R@1={recall:.4f}")
    fig.colorbar(image, ax=ax, label="Cosine Similarity")
    similarity_image = _figure_image(fig, path_for("similarity"))

    tactile_2d, force_2d = joint_umap_projection(
        tactile, force, random_seed=random_seed,
    )
    segments = matching_segments(tactile_2d, force_2d)
    fig, ax = plt.subplots(figsize=(8, 7))
    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], color="0.7", linewidth=0.5, alpha=0.6)
    ax.scatter(tactile_2d[:, 0], tactile_2d[:, 1], c="tab:blue", marker="o", label="Tactile")
    ax.scatter(force_2d[:, 0], force_2d[:, 1], c="tab:orange", marker="x", label="Force/Pressure")
    ax.set_title(f"Joint UMAP — {title_label}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend()
    umap_image = _figure_image(fig, path_for("umap"))

    distributions = similarity_distributions(similarity, random_seed=random_seed)
    bins = np.linspace(-1.0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(distributions["negative"], bins=bins, density=True, alpha=0.45, label="Negative")
    ax.hist(distributions["positive"], bins=bins, density=True, alpha=0.55, label="Positive")
    ax.hist(
        distributions["hard_negative"], bins=bins, density=True,
        histtype="step", linewidth=1.5, label="Hard Negative",
    )
    positive_mean = float(distributions["positive"].mean())
    negative_mean = float(distributions["negative"].mean())
    ax.axvline(positive_mean, color="tab:orange", linestyle="--", linewidth=1.2,
               label=f"Positive mean={positive_mean:.3f}")
    ax.axvline(negative_mean, color="tab:blue", linestyle="--", linewidth=1.2,
               label=f"Negative mean={negative_mean:.3f}")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Positive vs Negative Similarity — {title_label}")
    ax.legend(fontsize=8)
    distribution_image = _figure_image(fig, path_for("distribution"))

    images = {
        "alignment/visualization/similarity_matrix": similarity_image,
        "alignment/visualization/joint_umap": umap_image,
        "alignment/visualization/similarity_distribution": distribution_image,
    }
    return images, paths


def alignment_schedule(
    completed_epoch: int,
    *,
    retrieval_every_n_epochs: int,
    visualize_every_n_epochs: int,
) -> tuple[bool, bool]:
    """1-based completed epoch에서 retrieval/visualization 실행 여부를 반환한다."""
    if completed_epoch < 1:
        return False, False
    retrieve = completed_epoch % retrieval_every_n_epochs == 0
    visualize = completed_epoch % visualize_every_n_epochs == 0
    return retrieve or visualize, visualize


def should_run_final_alignment(
    *,
    enabled: bool,
    run_on_training_end: bool,
    current_step: int,
    last_evaluated_step: int,
) -> bool:
    """동일 model/global_step의 마지막 평가를 중복하지 않도록 판정한다."""
    return enabled and run_on_training_end and current_step != last_evaluated_step
