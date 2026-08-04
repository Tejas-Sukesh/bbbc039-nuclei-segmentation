"""Does the boundary sit where the image says the nucleus ends?

The metric conventions in `metrics.py` treat the hand-drawn mask as truth. At
loose IoU thresholds that is harmless, but mean AP@[.5:.95] spends over half of
its total shortfall on the 0.90 and 0.95 thresholds, where "correct" means
agreeing with the annotator to within roughly half a pixel. At that scale the
annotation is no longer obviously the more accurate of the two outlines, and
scoring against it stops measuring the model.

That is a claim about the ground truth, so it needs a referee that is neither
outline. The image is one. A nucleus has a real intensity edge, and the
conventional sub-pixel definition of its location is the **half-maximum**: the
level halfway between the object's interior and its local background. So for
each nucleus we can ask which of the two outlines lands closer to half-maximum,
and answer without ever consulting the other outline.

The comparison is deliberately **paired per object**: interior brightness,
background level and local contrast all vary hugely between nuclei and between
fields, and comparing pooled distributions would drown a half-pixel effect in
that variation. Comparing the two outlines on the *same* nucleus against the
*same* locally estimated reference cancels all of it -- the same reason the
exhaustive grid in `grids.py` beat the bandit that sampled random images.

Two statistics come out of it, per matched nucleus:

* `level` -- boundary intensity mapped onto the interior/background scale, so
  0.5 is exactly half-maximum. Above 0.5 the outline sits *inside* the true
  edge (traced tight); below 0.5 it sits *outside* (traced loose). This carries
  a direction, which is what makes it diagnostic rather than merely a score.
* `grad` -- gradient magnitude along the outline, normalised by local contrast.
  The steepest-descent definition of an edge rather than the half-maximum one;
  reported as an independent check that does not share the first one's
  assumption of a symmetric intensity profile.

What the numbers cannot do is separate annotator imprecision from a genuine
systematic difference in what a human and a network consider the edge of a
nucleus. Nobody traced any nucleus in BBBC039 twice, so inter-annotator
agreement -- the direct measure of human precision -- is not computable on this
dataset at all. The claim available here is the weaker but still decisive one:
*whichever outline is closer to the image's own evidence of where the nucleus
ends.*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import find_boundaries

from . import metrics as M

# A nucleus needs enough perimeter for a median over boundary pixels to mean
# anything, and enough contrast for the interior/background normalisation to be
# stable. Both cuts are on properties of the *image and ground truth only*, so
# neither can favour the prediction.
MIN_AREA = 150
MIN_CONTRAST = 100.0  # raw camera counts; the images span ~120-4095
PAD = 10  # crop margin, must exceed the background-ring radius
RING_RADIUS = 6
CORE_EROSION = 2


@dataclass(frozen=True)
class BoundaryFit:
    """Where each outline of one nucleus sits relative to its half-maximum."""

    name: str
    gt_id: int
    pred_id: int
    area: int
    contrast: float
    gt_level: float
    pred_level: float
    gt_grad: float
    pred_grad: float

    @property
    def gt_error(self) -> float:
        """Distance from half-maximum, in units of local contrast."""
        return abs(self.gt_level - 0.5)

    @property
    def pred_error(self) -> float:
        return abs(self.pred_level - 0.5)


def _disk(radius: int) -> np.ndarray:
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (yy**2 + xx**2) <= r * r


def _levels(
    image: np.ndarray, mask: np.ndarray, grad: np.ndarray, i_bg: float, contrast: float
) -> tuple[float, float]:
    """Median boundary intensity (on the half-max scale) and normalised gradient."""
    edge = find_boundaries(mask, mode="inner")
    if not edge.any():
        return float("nan"), float("nan")
    level = (float(np.median(image[edge])) - i_bg) / contrast
    return level, float(np.median(grad[edge])) / contrast


def fit_image(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    name: str = "",
    iou_threshold: float = 0.5,
) -> list[BoundaryFit]:
    """Measure both outlines against half-maximum for every matched nucleus."""
    gt, n_gt = M.relabel_sequential(gt)
    pred, n_pred = M.relabel_sequential(pred)
    if n_gt == 0 or n_pred == 0:
        return []

    matched = M.match_instances(M.iou_matrix(gt, pred), iou_threshold).matched
    # One lightly smoothed gradient field per image: Sobel on raw 16-bit data
    # tracks shot noise rather than the nucleus edge.
    grad = sobel(ndi.gaussian_filter(image.astype(np.float32), 1.0))
    ring_se, core_se = _disk(RING_RADIUS), _disk(CORE_EROSION)
    h, w = gt.shape
    fits = []

    objects = ndi.find_objects(gt)
    for gid, pid in matched:
        sl = objects[gid - 1]
        if sl is None:
            continue
        # Nuclei clipped by the field of view have an outline the annotator never
        # drew, so their boundary statistics are meaningless.
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w:
            continue
        y0, y1 = max(sl[0].start - PAD, 0), min(sl[0].stop + PAD, h)
        x0, x1 = max(sl[1].start - PAD, 0), min(sl[1].stop + PAD, w)
        img_c = image[y0:y1, x0:x1].astype(np.float32)
        gt_c, pred_c = gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]
        gt_m, pred_m = gt_c == gid, pred_c == pid

        area = int(gt_m.sum())
        if area < MIN_AREA:
            continue

        # Interior reference: pixels both outlines agree are inside, eroded away
        # from either boundary so neither outline's own error leaks into it.
        core = ndi.binary_erosion(gt_m & pred_m, core_se)
        both = gt_m | pred_m
        # Background reference: a ring around the union, minus anything either
        # labelling calls an object -- otherwise a touching neighbour's interior
        # is averaged in as "background".
        ring = ndi.binary_dilation(both, ring_se) & ~both & (gt_c == 0) & (pred_c == 0)
        if core.sum() < 20 or ring.sum() < 40:
            continue

        i_in, i_bg = float(np.median(img_c[core])), float(np.median(img_c[ring]))
        contrast = i_in - i_bg
        if contrast < MIN_CONTRAST:
            continue

        grad_c = grad[y0:y1, x0:x1]
        gt_level, gt_grad = _levels(img_c, gt_m, grad_c, i_bg, contrast)
        pred_level, pred_grad = _levels(img_c, pred_m, grad_c, i_bg, contrast)
        if not np.isfinite([gt_level, pred_level, gt_grad, pred_grad]).all():
            continue

        fits.append(
            BoundaryFit(
                name=name,
                gt_id=int(gid),
                pred_id=int(pid),
                area=area,
                contrast=contrast,
                gt_level=gt_level,
                pred_level=pred_level,
                gt_grad=gt_grad,
                pred_grad=pred_grad,
            )
        )
    return fits


def summarize(fits: list[BoundaryFit]) -> dict:
    """Paired comparison of the two outlines, with a signed-rank test."""
    if not fits:
        return {}
    gt_lvl = np.array([f.gt_level for f in fits])
    pr_lvl = np.array([f.pred_level for f in fits])
    gt_err = np.array([f.gt_error for f in fits])
    pr_err = np.array([f.pred_error for f in fits])
    gt_grd = np.array([f.gt_grad for f in fits])
    pr_grd = np.array([f.pred_grad for f in fits])

    out = {
        "n_objects": len(fits),
        "n_images": len({f.name for f in fits}),
        "gt_level_median": float(np.median(gt_lvl)),
        "pred_level_median": float(np.median(pr_lvl)),
        "gt_abs_error_median": float(np.median(gt_err)),
        "pred_abs_error_median": float(np.median(pr_err)),
        "pred_closer_fraction": float(np.mean(pr_err < gt_err)),
        "gt_grad_median": float(np.median(gt_grd)),
        "pred_grad_median": float(np.median(pr_grd)),
        "pred_sharper_fraction": float(np.mean(pr_grd > gt_grd)),
    }
    try:
        from scipy.stats import wilcoxon

        # Paired, two-sided: is either outline systematically closer to half-max?
        out["wilcoxon_p_abs_error"] = float(wilcoxon(pr_err, gt_err).pvalue)
        out["wilcoxon_p_gradient"] = float(wilcoxon(pr_grd, gt_grd).pvalue)
    except Exception:  # scipy.stats is optional for the rest of the module
        pass
    return out


def verdict(summary: dict) -> str:
    """One line stating what the paired comparison licenses us to say."""
    if not summary:
        return "no measurable objects"
    gt_e, pr_e = summary["gt_abs_error_median"], summary["pred_abs_error_median"]
    frac = summary["pred_closer_fraction"]
    p = summary.get("wilcoxon_p_abs_error", float("nan"))
    who = "the prediction" if pr_e < gt_e else "the ground truth"
    return (
        f"{who} sits closer to half-maximum on {max(frac, 1 - frac):.0%} of "
        f"{summary['n_objects']} nuclei "
        f"(median offset {min(pr_e, gt_e):.3f} vs {max(pr_e, gt_e):.3f} of local "
        f"contrast, Wilcoxon p={p:.2e})"
    )
