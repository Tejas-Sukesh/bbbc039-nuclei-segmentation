"""Figures for the failure analysis.

Conventions worth stating, because each is a choice that hides errors if made the
other way:

* **Boundaries, not filled masks.** A filled overlay hides exactly the boundary
  disagreements that drive AP at the strict IoU thresholds. Outlines make them
  visible.
* **Random-permutation colormap for label images.** A sequential colormap gives
  adjacent instance IDs near-identical colours, which is precisely how a split or
  a merge becomes invisible in a figure.
* **One colour language across every figure**: orange is always the *human*
  annotation, blue is always the *model* prediction, in the charts and in the
  image overlays alike. A reader should never have to re-learn the legend, and
  the two hues are separable under deuteranopia and tritanopia (worst-pair CVD
  dE 9.2, normal-vision dE 24.0 on this surface).
* **Text never wears a series colour.** Values and labels stay in ink; a coloured
  mark beside them carries the identity.
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

# Validated categorical slots (light surface). GT/human = orange, model = blue.
C_GT = "#eb6834"
C_PRED = "#2a78d6"
C_ACCENT = "#4a3aa7"  # third series / spurious objects
C_INK = "#0b0b0b"
C_INK_SOFT = "#52514e"
C_SURFACE = "#ffffff"
C_GRID = "#d8d7d2"

# Brighter steps of the same two hues, for outlines drawn over greyscale pixels
# rather than over the chart surface.
C_GT_OVERLAY = (1.00, 0.55, 0.20)
C_PRED_OVERLAY = (0.30, 0.70, 1.00)


def _style(ax) -> None:
    """Recessive axes and grid, so the marks carry the figure."""
    ax.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_INK_SOFT, labelsize=9, length=3)
    ax.grid(alpha=0.35, color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=C_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


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


def outlined(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Greyscale image as RGB with both outlines drawn on it."""
    rgb = np.dstack([normalize_for_display(image)] * 3)
    rgb[find_boundaries(gt, mode="outer")] = C_GT_OVERLAY
    rgb[find_boundaries(pred, mode="outer")] = C_PRED_OVERLAY
    return np.clip(rgb, 0, 1)


def overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out: Path,
            title: str = "") -> Path:
    """Raw image with hand-drawn outlines in orange and predicted ones in blue."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(outlined(image, gt, pred))
    ax.set_title(f"{title}\norange = hand-drawn ground truth, blue = prediction",
                 fontsize=10, color=C_INK)
    ax.axis("off")
    return _save(fig, out)


def error_map(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """RGB map colour-coding matched / missed / spurious objects.

    green  = matched above threshold
    red    = ground-truth nucleus with no match (missed)
    violet = predicted object with no match (spurious)

    Violet rather than blue for the spurious class: blue means "the model's
    outline" everywhere else in these figures, and reusing it for a category
    would make the legend context-dependent.
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
            out[pred_r == p] = (0.29, 0.23, 0.65)
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
    axes[3].set_title("green=matched  red=missed  violet=spurious")
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
    _style(ax)
    ax.hist(aps, bins=20, color=C_PRED, edgecolor=C_SURFACE, linewidth=1.2)
    ax.axvline(float(np.mean(aps)), color=C_INK, ls="--", linewidth=1.5)
    ax.annotate(f"mean {np.mean(aps):.3f}", xy=(float(np.mean(aps)), ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", fontsize=9, color=C_INK)
    ax.set_xlabel("per-image AP@[.5:.95]", color=C_INK_SOFT)
    ax.set_ylabel("images", color=C_INK_SOFT)
    ax.set_title("Per-image score spread: the mean hides a 0.70-0.90 range",
                 fontsize=11, color=C_INK, loc="left")
    return _save(fig, out)


# --------------------------------------------------------------------------- #
# Where the score is lost, and to what
# --------------------------------------------------------------------------- #


def ap_by_threshold(per_threshold: dict, out: Path) -> Path:
    """Score against IoU threshold, beside each threshold's share of the shortfall.

    Two measures on one x axis, so they get two panels rather than two y scales.
    The right panel is the one that reorders the priorities: it shows how much of
    the total gap to a perfect score each threshold is responsible for, and the
    strictest two dominate it.
    """
    ts = sorted(float(t) for t in per_threshold)
    vals = np.array([per_threshold[t] if t in per_threshold else per_threshold[str(t)]
                     for t in ts])
    shortfall = 1.0 - vals
    share = shortfall / shortfall.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax in (ax1, ax2):
        _style(ax)

    ax1.plot(ts, vals, color=C_PRED, linewidth=2, marker="o", markersize=6,
             markeredgecolor=C_SURFACE, markeredgewidth=1.2, zorder=3)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("IoU threshold for calling a nucleus correct", color=C_INK_SOFT)
    ax1.set_ylabel("score at that threshold", color=C_INK_SOFT)
    ax1.set_title(f"Mean AP@[.5:.95] = {vals.mean():.3f} is an average of these",
                  fontsize=11, color=C_INK, loc="left")
    for t in (ts[0], 0.85, ts[-1]):
        if t in ts:
            v = vals[ts.index(t)]
            ax1.annotate(f"{v:.2f}", xy=(t, v), xytext=(0, 10),
                         textcoords="offset points", ha="center", fontsize=9,
                         color=C_INK, fontweight="bold")

    bars = ax2.bar(ts, share, width=0.035, color=C_PRED, zorder=3)
    worst = int(np.argmax(share))
    for i, b in enumerate(bars):
        if share[i] < 0.12 and i != worst:
            continue
        ax2.annotate(f"{share[i]:.0%}", xy=(b.get_x() + b.get_width() / 2, share[i]),
                     xytext=(0, 4), textcoords="offset points", ha="center",
                     fontsize=9, color=C_INK, fontweight="bold")
    tail = share[-2:].sum()
    ax2.set_xlabel("IoU threshold", color=C_INK_SOFT)
    ax2.set_ylabel("share of the total shortfall", color=C_INK_SOFT)
    ax2.set_title(f"The two strictest thresholds cause {tail:.0%} of the loss",
                  fontsize=11, color=C_INK, loc="left")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return _save(fig, out)


def annotation_ceiling(fits_csv: Path, out: Path, summary: dict | None = None) -> Path:
    """Where each outline sits relative to the image's own edge (half-maximum).

    The single most important figure in the writeup: it is the evidence that at
    strict IoU thresholds the model is being penalised against an outline that is
    itself further from the true edge than the model's is.
    """
    with Path(fits_csv).open() as fh:
        rows = list(csv.DictReader(fh))
    gt = np.array([float(r["gt_level"]) for r in rows])
    pr = np.array([float(r["pred_level"]) for r in rows])
    keep = np.isfinite(gt) & np.isfinite(pr) & (np.abs(gt - 0.5) < 0.5) & (np.abs(pr - 0.5) < 0.5)
    gt, pr = gt[keep], pr[keep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (ax1, ax2):
        _style(ax)

    bins = np.linspace(0.15, 0.85, 60)
    ax1.hist(gt, bins=bins, color=C_GT, alpha=0.85, label="hand-drawn ground truth",
             edgecolor=C_SURFACE, linewidth=0.4, zorder=3)
    ax1.hist(pr, bins=bins, color=C_PRED, alpha=0.75, label="model prediction",
             edgecolor=C_SURFACE, linewidth=0.4, zorder=3)
    ax1.axvline(0.5, color=C_INK, linewidth=1.6, zorder=4)
    ax1.annotate("true edge\n(half-maximum)", xy=(0.5, ax1.get_ylim()[1] * 0.97),
                 xytext=(8, -4), textcoords="offset points", fontsize=9,
                 color=C_INK, va="top")
    ax1.set_xlabel("where the outline sits on the brightness ramp"
                   "  (0.5 = half-maximum)", color=C_INK_SOFT)
    ax1.set_ylabel("nuclei", color=C_INK_SOFT)
    ax1.set_title(f"Both outlines read below half-maximum, the model further"
                  f"  (n={len(gt)} nuclei)", fontsize=11, color=C_INK, loc="left")
    # Upper left: the sparse tail of both distributions, and the only corner the
    # "true edge" annotation at x=0.5 cannot collide with.
    leg = ax1.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(C_INK)

    # The right panel is the result: two defensible edge definitions rank the two
    # outlines oppositely. A single "who won" bar would hide exactly that.
    img = (summary or {}).get("per_image", {})
    lvl_pred = img.get("pred_closer_fraction", float("nan"))
    grd_pred = img.get("pred_sharper_fraction", float("nan"))
    n_img = img.get("n_images", 0)
    labels = ["half-maximum\nlevel", "gradient\nmagnitude"]
    pred_share = np.array([lvl_pred, grd_pred])
    gt_share = 1.0 - pred_share
    x = np.arange(len(labels))
    ax2.bar(x, gt_share, width=0.55, color=C_GT, zorder=3, label="favours ground truth")
    ax2.bar(x, pred_share, width=0.55, bottom=gt_share, color=C_PRED, zorder=3,
            label="favours prediction")
    for i, (g, p) in enumerate(zip(gt_share, pred_share)):
        ax2.annotate(f"{g:.0%}", xy=(i, g / 2), ha="center", va="center",
                     fontsize=11, color="white", fontweight="bold")
        ax2.annotate(f"{p:.0%}", xy=(i, g + p / 2), ha="center", va="center",
                     fontsize=11, color="white", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.set_ylabel(f"share of {n_img} fields", color=C_INK_SOFT)
    ax2.set_title("Only the gradient measure is well-posed\n"
                  "half-maximum needs a plateau these nuclei do not have",
                  fontsize=11, color=C_INK, loc="left")
    ax2.annotate("not evidence", xy=(0, 1.0), xytext=(0, 6),
                 textcoords="offset points", ha="center", fontsize=8.5,
                 color=C_INK_SOFT, style="italic")
    leg = ax2.legend(frameon=False, fontsize=8.5, loc="lower center",
                     bbox_to_anchor=(0.5, -0.32), ncol=2)
    for t in leg.get_texts():
        t.set_color(C_INK)
    return _save(fig, out)


def what_gets_missed(missed_csv: Path, gt_areas: np.ndarray, out: Path) -> Path:
    """Missed nuclei by mechanism and by size, against the size of all nuclei.

    The size panels are the argument that the misses are not the hard touching
    cases people expect but a population of very small objects.
    """
    with Path(missed_csv).open() as fh:
        rows = list(csv.DictReader(fh))
    area = np.array([float(r["area"]) for r in rows])
    kinds = [r["kind"] for r in rows]
    order = ["undetected", "absorbed", "boundary"]
    labels = {
        "undetected": "never\ndetected",
        "absorbed": "absorbed by\na neighbour",
        "boundary": "outline\ndrifted",
    }
    counts = [kinds.count(k) for k in order]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for ax in axes:
        _style(ax)

    bars = axes[0].bar([labels[k] for k in order], counts, width=0.55,
                       color=[C_PRED, C_GT, C_ACCENT], zorder=3)
    for b, c in zip(bars, counts):
        axes[0].annotate(f"{c}  ({c/len(rows):.0%})",
                         xy=(b.get_x() + b.get_width() / 2, c), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=9.5,
                         color=C_INK, fontweight="bold")
    axes[0].set_ylabel("nuclei missed", color=C_INK_SOFT)
    axes[0].set_ylim(0, max(counts) * 1.22)
    axes[0].set_title(f"{len(rows)} missed nuclei, by mechanism",
                      fontsize=11, color=C_INK, loc="left")

    # Each distribution scaled to its own peak: the two populations differ ~16x
    # in count, and a shared density axis flattens the reference one into the
    # axis line. The comparison being made here is of shape, not of frequency.
    bins = np.logspace(0, np.log10(max(gt_areas.max(), area.max())), 40)
    for data, colour, label in ((gt_areas, C_ACCENT, "all nuclei"),
                                (area, C_GT, "missed nuclei")):
        h, _ = np.histogram(data, bins=bins)
        axes[1].stairs(h / h.max(), bins, fill=True, alpha=0.6, color=colour,
                       label=label, zorder=3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("nucleus area (pixels, log scale)", color=C_INK_SOFT)
    axes[1].set_ylabel("nuclei (each curve scaled to its own peak)",
                       color=C_INK_SOFT, fontsize=8.5)
    axes[1].set_title(f"Missed nuclei are tiny: median {np.median(area):.0f} px"
                      f" vs {np.median(gt_areas):.0f} px",
                      fontsize=11, color=C_INK, loc="left")
    leg = axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(C_INK)

    edges = np.array([0, 15, 50, 100, 200, 400, 800, 1e9])
    names = ["<15", "15-50", "50-100", "100-200", "200-400", "400-800", ">800"]
    frac, tot = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n_all = int(((gt_areas >= lo) & (gt_areas < hi)).sum())
        n_bad = int(((area >= lo) & (area < hi)).sum())
        frac.append(n_bad / n_all if n_all else 0.0)
        tot.append(n_all)
    bars = axes[2].bar(names, frac, width=0.62, color=C_GT, zorder=3)
    for b, f, n in zip(bars, frac, tot):
        axes[2].annotate(f"{f:.0%}\nof {n}", xy=(b.get_x() + b.get_width() / 2, f),
                         xytext=(0, 3), textcoords="offset points", ha="center",
                         fontsize=8, color=C_INK)
    axes[2].set_xlabel("nucleus area (pixels)", color=C_INK_SOFT)
    axes[2].set_ylabel("fraction missed", color=C_INK_SOFT)
    axes[2].set_ylim(0, 1.22)
    axes[2].tick_params(axis="x", labelsize=8, rotation=30)
    axes[2].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[2].set_title("Detection collapses below ~50 px", fontsize=11,
                      color=C_INK, loc="left")
    return _save(fig, out)


def missed_gallery(rows, predict, out: Path, pad: int = 26, zoom: int = 5) -> Path:
    """Close-ups of missed objects across the size range, for eyeballing.

    The failure analysis says 81% of misses are under 100 px and the smallest is
    2 px. Whether that is a detection failure or a disagreement about what counts
    as a nucleus is not something a table can settle -- it needs looking at. This
    renders the raw pixels around each missed object at high zoom, with the
    annotation drawn on, so the question "is that actually a nucleus?" can be
    answered by inspection rather than asserted.

    `rows` should span the size range, not just the smallest, so the reader can
    see where the population stops looking like debris and starts looking like
    cells.
    """
    rows = list(rows)
    cols = min(4, len(rows))
    nrow = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nrow, cols, figsize=(3.5 * cols, 3.9 * nrow),
                             squeeze=False)
    cache: dict[str, tuple] = {}
    for ax, r in zip(axes.ravel(), rows):
        name = r["name"]
        if name not in cache:
            s = load_sample(name)
            cache[name] = (s.image, M.relabel_sequential(s.labels)[0], predict(name))
        image, gt, pred = cache[name]
        gid = int(r["gt_id"])
        ys, xs = np.nonzero(gt == gid)
        if not len(ys):
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, y1 = max(cy - pad, 0), min(cy + pad, image.shape[0])
        x0, x1 = max(cx - pad, 0), min(cx + pad, image.shape[1])
        sub = outlined(image[y0:y1, x0:x1], (gt[y0:y1, x0:x1] == gid).astype(np.int32),
                       pred[y0:y1, x0:x1])
        ax.imshow(np.kron(sub, np.ones((zoom, zoom, 1))), interpolation="nearest")
        ax.set_title(f"{float(r['area']):.0f} px  ·  brightness {float(r['mean_intensity']):.0f}"
                     f"\n{r['kind']}", fontsize=9, color=C_INK)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Missed objects, smallest to largest.  orange = what the human marked,"
                 "  blue = what the model found\n"
                 "The question each panel asks: is that a nucleus, or a speck?",
                 fontsize=11, color=C_INK)
    return _save(fig, out)


def merge_gallery(merges, predict, out: Path, n: int = 6, pad: int = 18,
                  subtitle: str = "") -> Path:
    """Close-ups of the predictions that fused several nuclei.

    Cropped rather than whole-field: at 520x696 a fused pair is a few dozen
    pixels and simply invisible in a full-frame figure. Each panel is labelled
    with the *kind* of merge it is, because the mix is the finding -- most of
    these are a normal nucleus plus a small bright punctum the annotator called
    its own nucleus, not two nuclei of similar size that the network failed to
    separate.
    """
    merges = list(merges)[:n]
    if not merges:
        raise ValueError("no merged predictions to render")
    cols = min(3, len(merges))
    rows = int(np.ceil(len(merges) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows),
                             squeeze=False)
    cache: dict[str, tuple] = {}
    for ax, m in zip(axes.ravel(), merges):
        if m.name not in cache:
            s = load_sample(m.name)
            cache[m.name] = (s.image, s.labels, predict(m.name))
        image, gt, pred = cache[m.name]
        y0, y1, x0, x1 = m.bbox
        y0, y1 = max(y0 - pad, 0), min(y1 + pad, image.shape[0])
        x0, x1 = max(x0 - pad, 0), min(x1 + pad, image.shape[1])
        ax.imshow(outlined(image[y0:y1, x0:x1], gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]),
                  interpolation="nearest")
        kind = getattr(m, "kind", "")
        areas = " + ".join(str(a) for a in getattr(m, "gt_areas", ())) or "?"
        ax.set_title(f"{m.n_absorbed} annotated nuclei -> 1 prediction   [{kind}]\n"
                     f"areas {areas} px", fontsize=9, color=C_INK)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Merges: orange = hand-drawn nuclei, blue = the single object predicted"
                 f" over them{chr(10) + subtitle if subtitle else ''}",
                 fontsize=11, color=C_INK)
    return _save(fig, out)
