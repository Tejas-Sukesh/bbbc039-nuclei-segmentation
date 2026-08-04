"""Figures for the failure analysis.

Two conventions worth stating, because both are choices that hide errors if made
the other way:

* **Boundaries, not filled masks.** A filled overlay hides exactly the boundary
  disagreements that drive AP at the strict IoU thresholds. Outlines make them
  visible.
* **Random-permutation colormap for label images.** A sequential colormap gives
  adjacent instance IDs near-identical colours, which is precisely how a split or
  a merge becomes invisible in a figure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries

from . import metrics as M
from .data import load_sample


def normalize_for_display(image: np.ndarray, p_low=1.0, p_high=99.8) -> np.ndarray:
    lo, hi = np.percentile(image, [p_low, p_high])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def label_cmap(n: int = 512, seed: int = 0) -> ListedColormap:
    """Random-permutation colormap so neighbouring IDs are visually distinct."""
    rng = np.random.default_rng(seed)
    colors = rng.random((max(n, 2), 3)) * 0.75 + 0.25
    colors[0] = (0, 0, 0)  # background
    return ListedColormap(colors)


def overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out: Path,
            title: str = "") -> Path:
    """Raw image with GT outlines in green and predicted outlines in magenta."""
    disp = normalize_for_display(image)
    rgb = np.dstack([disp] * 3)
    rgb[find_boundaries(gt, mode="outer")] = (0.1, 1.0, 0.2)
    rgb[find_boundaries(pred, mode="outer")] = (1.0, 0.1, 0.9)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_title(f"{title}\ngreen = ground truth, magenta = prediction", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def error_map(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """RGB map colour-coding matched / missed / spurious objects.

    green  = matched above threshold
    red    = ground-truth nucleus with no match (missed)
    blue   = predicted object with no match (spurious)
    """
    gt_r, n_gt = M.relabel_sequential(gt)
    pred_r, n_pred = M.relabel_sequential(pred)
    out = np.zeros((*gt.shape, 3), dtype=np.float32)
    if n_gt == 0 and n_pred == 0:
        return out

    ious = M.iou_matrix(gt_r, pred_r)
    m = M.match_instances(ious, iou_threshold) if ious.size else None
    matched_gt = {g for g, _ in (m.matched if m else [])}
    matched_pred = {p for _, p in (m.matched if m else [])}

    for g in range(1, n_gt + 1):
        mask = gt_r == g
        out[mask] = (0.15, 0.75, 0.25) if g in matched_gt else (0.9, 0.15, 0.15)
    for p in range(1, n_pred + 1):
        if p not in matched_pred:
            out[pred_r == p] = (0.2, 0.35, 0.95)
    return out


def error_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out: Path,
                title: str = "") -> Path:
    """Four-up: raw / ground truth / prediction / error map."""
    cmap = label_cmap()
    score = M.score_image(gt, pred)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(normalize_for_display(image), cmap="gray")
    axes[0].set_title("raw")
    axes[1].imshow(gt % cmap.N, cmap=cmap, interpolation="nearest")
    axes[1].set_title(f"ground truth ({score.n_gt} nuclei)")
    axes[2].imshow(pred % cmap.N, cmap=cmap, interpolation="nearest")
    axes[2].set_title(f"prediction ({score.n_pred} nuclei)")
    axes[3].imshow(error_map(gt, pred))
    axes[3].set_title("green=matched  red=missed  blue=spurious")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{title}   AP={score.ap:.3f}  F1@0.5={score.f1_50:.3f}  "
        f"count {score.n_pred}/{score.n_gt} ({score.count_error:+d})  "
        f"splits={score.splits} merges={score.merges}",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def worst_cases(per_image_csv: Path, predict, n: int, out_dir: Path) -> list[Path]:
    """Render error panels for the N lowest-scoring fields.

    The failure analysis should be built from these rather than from
    cherry-picked examples, which is the whole reason `evaluate` writes per-image
    rows in the first place.
    """
    with Path(per_image_csv).open() as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: float(r["ap"]))
    made = []
    for rank, row in enumerate(rows[:n], 1):
        sample = load_sample(row["name"])
        made.append(
            error_panel(
                sample.image,
                sample.labels,
                predict(row["name"]),
                out_dir / f"worst_{rank:02d}_{row['name'][:20]}.png",
                title=f"rank {rank} worst  ({row['name'][:26]})",
            )
        )
    return made


def stage_panel(image: np.ndarray, out: Path, params=None) -> Path:
    """Intermediate products of the classical pipeline, to attribute an error to a stage."""
    from .baseline import BaselineParams, find_seeds, foreground_mask, normalize, segment

    params = params or BaselineParams()
    norm = normalize(image, params)
    mask = foreground_mask(norm, params)
    markers, edt = find_seeds(mask, params)
    labels = segment(image, params)

    cmap = label_cmap()
    panels = [
        (norm, "1. normalized", "gray"),
        (mask, "2. foreground (Otsu)", "gray"),
        (edt, "3. distance transform", "magma"),
        (markers > 0, f"4. seeds ({markers.max()})", "gray"),
        (labels % cmap.N, f"5. watershed ({labels.max()})", cmap),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 5))
    for ax, (data, name, cm) in zip(axes, panels):
        ax.imshow(data, cmap=cm, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def sweep_curve(per_arm_ap: dict[str, float], out: Path, param: str = "") -> Path:
    """Metric against configuration, sorted, for the before/after section."""
    items = sorted(per_arm_ap.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(items))))
    ax.barh(range(len(vals)), vals, color="#4477aa")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7, family="monospace")
    ax.set_xlim(min(vals) - 0.01, max(vals) + 0.005)
    ax.set_xlabel("mean AP@[.5:.95]")
    ax.set_title(f"parameter sweep{f' ({param})' if param else ''}", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def score_histogram(per_image_csv: Path, out: Path) -> Path:
    """Distribution of per-image AP -- shows what a single mean hides."""
    with Path(per_image_csv).open() as fh:
        aps = [float(r["ap"]) for r in csv.DictReader(fh)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(aps, bins=20, color="#4477aa", edgecolor="white")
    ax.axvline(float(np.mean(aps)), color="crimson", ls="--",
               label=f"mean {np.mean(aps):.3f}")
    ax.set_xlabel("per-image AP@[.5:.95]")
    ax.set_ylabel("images")
    ax.legend()
    ax.set_title("per-image score distribution", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
