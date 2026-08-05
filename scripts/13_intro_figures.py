#!/usr/bin/env python
"""Two orientation figures for an audience new to this problem.

  slideA_why_instances.png   foreground vs instances: why "which pixels are
                             nuclei" is not the question anyone actually asks
  slideB_otsu_vs_cellpose.png  the same field through the classical pipeline
                             and through Cellpose-SAM, against the annotation

Both render real pipeline output, not diagrams, so every count on the slide is
something the repo actually produced.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from nucleiseg import viz
from nucleiseg.data import REPO_ROOT, load_sample
from nucleiseg.segmenters import CellposeParams, ClassicalSegmenter, FlowCache

FIGURES = REPO_ROOT / "figures"
DPI = 200
FIELD = "IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86"

CROP_H, CROP_W = 210, 300


def pick_crop(gt, classical, step=40):
    """The window where the classical pipeline splits the most GT nuclei.

    Chosen by scanning every candidate window and sorting, never by eye, so the
    teaching example is the pipeline's characteristic failure rather than the
    one that happened to look convincing.
    """
    h, w = gt.shape
    best, best_score = None, -1
    for r in range(0, h - CROP_H + 1, step):
        for c in range(0, w - CROP_W + 1, step):
            sl = (slice(r, r + CROP_H), slice(c, c + CROP_W))
            g, p = gt[sl], classical[sl]
            splits = 0
            for i in np.unique(g):
                if i == 0:
                    continue
                m = g == i
                if m.sum() < 60:
                    continue
                # Predicted pieces that each claim a real share of this nucleus.
                parts = [q for q in np.unique(p[m]) if q != 0
                         and (p[m] == q).sum() > 0.15 * m.sum()]
                if len(parts) >= 2:
                    splits += 1
            if splits > best_score:
                best, best_score = sl, splits
    print(f"  crop splits={best_score}")
    return best


def panel(ax, rgb, title, sub=None, count=None):
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(viz.C_GRID)
    ax.set_title(title, fontsize=13.5, color=viz.C_INK, pad=7)
    if sub:
        ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color=viz.C_INK_SOFT)
    if count is not None:
        ax.text(0.03, 0.96, count, transform=ax.transAxes, ha="left", va="top",
                fontsize=19, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000",
                          alpha=0.66, edgecolor="none"))


def slideA_why_instances(s, crop) -> None:
    """Foreground mask vs instance labels, on a crowded crop.

    The middle panel is a real Otsu threshold, not the annotation with the IDs
    thrown away -- the claim is that thresholding is genuinely easy, so it has
    to be an actual threshold.
    """
    from nucleiseg.baseline import BaselineParams, foreground_mask, normalize

    p = BaselineParams()
    thresh = foreground_mask(normalize(s.image, p), p)[crop]

    lab = s.labels[crop]
    disp = viz.normalize_for_display(s.image[crop])

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))

    panel(axes[0], np.dstack([disp] * 3), "the image",
          "one field of stained nuclei")

    fg = np.dstack([disp] * 3)
    fg[thresh] = (0.18, 0.47, 0.84)
    panel(axes[1], fg, "foreground: which pixels are nuclei",
          "one line of Otsu — and not the question anyone asks")

    cmap = viz.label_cmap(n=256, seed=3)
    inst = np.zeros((*lab.shape, 3))
    inst[..., :] = np.dstack([disp * 0.30] * 3)
    for i in np.unique(lab):
        if i == 0:
            continue
        inst[lab == i] = cmap(int(i) % cmap.N)[:3]
    panel(axes[2], inst, "instances: which nucleus is which",
          f"the actual task — {len(np.unique(lab)) - 1} separate objects here")

    fig.suptitle("Counting cells means separating them, not just finding them",
                 fontsize=16, color=viz.C_INK)
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.02, right=0.98, wspace=0.06)
    fig.savefig(FIGURES / "slideA_why_instances.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def slideB_otsu_vs_cellpose(s, classical, cellpose, crop) -> None:
    """Annotation, classical pipeline, Cellpose-SAM — same crop, outlines only.

    Outlines rather than filled masks: a fill hides the boundary disagreements,
    and boundary behaviour is the whole difference between these two.
    """
    disp = viz.normalize_for_display(s.image[crop])

    def outlined(lab, colour):
        rgb = np.dstack([disp] * 3)
        rgb[find_boundaries(lab, mode="outer")] = colour
        return rgb

    gt = s.labels[crop]
    cl = classical[crop]
    cp = cellpose[crop]

    def n(a):
        return len(np.unique(a)) - (1 if 0 in a else 0)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    panel(axes[0], outlined(gt, viz.C_GT_OVERLAY), "what a human drew",
          "the annotation", f"{n(gt)}")
    panel(axes[1], outlined(cl, (0.62, 0.40, 0.90)),
          "Otsu threshold → watershed",
          "classical, no learning — mean AP 0.555", f"{n(cl)}")
    panel(axes[2], outlined(cp, viz.C_PRED_OVERLAY), "Cellpose-SAM",
          "pretrained, zero-shot — mean AP 0.807", f"{n(cp)}")

    fig.suptitle("The same crowded corner, three ways",
                 fontsize=16, color=viz.C_INK)
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.02, right=0.98, wspace=0.06)
    fig.savefig(FIGURES / "slideB_otsu_vs_cellpose.png", dpi=DPI, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    s = load_sample(FIELD)
    print(f"field {FIELD}: {s.n_nuclei} nuclei")

    print("classical ...")
    classical = ClassicalSegmenter().segment(s.image)
    print("cellpose ...")
    cellpose = FlowCache().masks(FIELD, CellposeParams())

    crop = pick_crop(s.labels, classical)
    slideA_why_instances(s, crop)
    print("  slideA_why_instances.png")
    slideB_otsu_vs_cellpose(s, classical, cellpose, crop)
    print("  slideB_otsu_vs_cellpose.png")
