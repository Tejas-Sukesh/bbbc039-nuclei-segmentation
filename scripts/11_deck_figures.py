#!/usr/bin/env python
"""Render the three figures the slide deck needs that the analysis did not produce.

  slide2_the_task.png   raw field | the same field with human outlines
  slide3_the_trap.png   the mask read the obvious way | decoded correctly
  slide4_what_i_built.png   four-box pipeline diagram

Uses the project's own colour language from nucleiseg.viz so the new slides sit
beside fig1..fig6 without a visual seam: orange = human, blue = model.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from skimage.segmentation import find_boundaries

from nucleiseg import viz
from nucleiseg.data import MASKS, REPO_ROOT, decode_mask, load_sample, split_names

FIGURES = REPO_ROOT / "figures"
DPI = 200


def pick_field(target: int = 118) -> str:
    """The validation field whose nucleus count is closest to the dataset mean.

    Chosen by sorting on |count - target|, not by eye, so the illustrative
    example is not cherry-picked to flatter the point.
    """
    best, best_d = None, 10**9
    for name in split_names("validation"):
        n = int(decode_mask(MASKS / f"{name}.png").max())
        if abs(n - target) < best_d:
            best, best_d = name, abs(n - target)
    return best


def slide2_the_task(name: str) -> None:
    s = load_sample(name)
    disp = viz.normalize_for_display(s.image)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    axes[0].imshow(disp, cmap="gray", interpolation="nearest")
    axes[0].set_title("what the microscope gives you", fontsize=15, color=viz.C_INK)

    rgb = np.dstack([disp] * 3)
    edges = find_boundaries(s.labels, mode="outer")
    rgb[edges] = viz.C_GT_OVERLAY
    axes[1].imshow(rgb, interpolation="nearest")
    axes[1].set_title(
        f"what you have to produce  ({s.n_nuclei} separate objects)",
        fontsize=15, color=viz.C_INK,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(viz.C_GRID)
    fig.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98, wspace=0.06)
    fig.savefig(FIGURES / "slide2_the_task.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def slide3_the_trap(name: str) -> None:
    """The same mask file under three readings.

    Two distinct mistakes, not one. Taking the red channel as instance IDs is
    the trap the file's appearance invites and collapses the field to 3
    objects. Binarising and running connected components is the more careful
    next guess, and it still fuses every touching pair. Only per-colour
    decoding recovers the annotation.
    """
    from scipy import ndimage as ndi

    raw = iio.imread(MASKS / f"{name}.png")
    coloring = (raw[..., 0] if raw.ndim == 3 else raw).astype(np.int32)

    as_ids = coloring                                    # pixel value == object ID
    n_ids = int(as_ids.max())
    binary, n_binary = ndi.label(coloring > 0)           # foreground components
    correct = decode_mask(MASKS / f"{name}.png")
    n_correct = int(correct.max())

    cmap = viz.label_cmap(n=max(n_binary, n_correct) + 8, seed=0)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))

    panels = (
        (as_ids, n_ids, "pixel value = object ID",
         "what the file looks like it is", True),
        (binary, n_binary, "binarise, then label",
         f"the careful next guess — still loses {n_correct - n_binary} nuclei", False),
        (correct, n_correct, "label within each colour",
         "the decoding this dataset requires", False),
    )

    for ax, (lab, n, title, sub, is_ids) in zip(axes, panels):
        if is_ids:
            # Three flat colours, so the graph colouring itself is visible.
            shown = np.ma.masked_equal(lab, 0)
            ax.imshow(shown, cmap=ListedColormap(["#eb6834", "#2a78d6", "#4a3aa7"]),
                      interpolation="nearest", vmin=1, vmax=3)
        else:
            shown = np.ma.masked_equal(np.where(lab > 0, lab % cmap.N, 0), 0)
            ax.imshow(shown, cmap=cmap, interpolation="nearest", vmin=0, vmax=cmap.N)
        ax.set_facecolor("#111111")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(viz.C_GRID)
        ax.set_title(title, fontsize=13, color=viz.C_INK, pad=7)
        ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color=viz.C_INK_SOFT)
        ax.text(0.035, 0.955, f"{n}", transform=ax.transAxes,
                ha="left", va="top", fontsize=34, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#000000",
                          alpha=0.66, edgecolor="none"))

    fig.suptitle("The same mask file, read three ways",
                 fontsize=17, color=viz.C_INK)
    fig.subplots_adjust(top=0.83, bottom=0.11, left=0.02, right=0.98, wspace=0.06)
    fig.savefig(FIGURES / "slide3_the_trap.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def slide4_what_i_built() -> None:
    boxes = [
        ("Ground-truth\ndecoder", "recovers 23,617\nnuclei from 200 masks"),
        ("Segmenter", "pretrained Cellpose\n+ classical from scratch"),
        ("Caching layer", "8.8 s → 0.11 s per image\nidentical output"),
        ("Measurement\nsuite", "AP, splits, merges,\nmisses, boundary fit"),
    ]
    fig, ax = plt.subplots(figsize=(13.2, 3.5))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 26)
    ax.axis("off")

    w, h, gap = 20.5, 13.0, 5.8
    x0, y0 = 2.0, 7.0
    fills = ["#f4f3ef", "#f4f3ef", "#e8f1fc", "#f4f3ef"]
    edges = [viz.C_GRID, viz.C_GRID, viz.C_PRED, viz.C_GRID]

    for i, ((title, sub), fc, ec) in enumerate(zip(boxes, fills, edges)):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=2.0 if i == 2 else 1.3,
        ))
        ax.text(x + w / 2, y0 + h * 0.66, title, ha="center", va="center",
                fontsize=14.5, fontweight="bold", color=viz.C_INK)
        ax.text(x + w / 2, y0 + h * 0.26, sub, ha="center", va="center",
                fontsize=10.5, color=viz.C_INK_SOFT, linespacing=1.5)
        if i < 3:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.9, y0 + h / 2), (x + w + gap - 0.9, y0 + h / 2),
                arrowstyle="-|>", mutation_scale=17,
                color=viz.C_INK_SOFT, linewidth=1.6,
            ))

    ax.text(x0 + 2 * (w + gap) + w / 2, y0 - 2.6, "72× — everything downstream depends on this",
            ha="center", va="top", fontsize=12, fontweight="bold", color=viz.C_PRED)
    fig.tight_layout()
    fig.savefig(FIGURES / "slide4_what_i_built.png", dpi=DPI, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    name = pick_field()
    n = int(decode_mask(MASKS / f"{name}.png").max())
    print(f"field: {name}  ({n} nuclei)")
    slide2_the_task(name)
    print("  slide2_the_task.png")
    slide3_the_trap(name)
    print("  slide3_the_trap.png")
    slide4_what_i_built()
    print("  slide4_what_i_built.png")
