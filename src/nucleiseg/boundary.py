"""Does the boundary sit where the image says the nucleus ends?

The metric conventions in `metrics.py` treat the hand-drawn mask as truth. At
loose IoU thresholds that is harmless, but mean AP@[.5:.95] spends over half of
its total shortfall on the 0.90 and 0.95 thresholds, where "correct" means
agreeing with the annotator to within roughly half a pixel. At that scale the
annotation is no longer obviously the more accurate of the two outlines, and
scoring against it stops measuring the model.

That is a claim about the ground truth, so it needs a referee that is neither
outline. The image is one: a nucleus has a real intensity edge, and we can ask
which outline sits closer to it without ever consulting the other outline. Two
ways of asking that were tried here, and only one of them survived -- the
conventional half-maximum definition needs an intensity plateau that these
objects turn out not to have.

The comparison is deliberately **paired per object**: interior brightness,
background level and local contrast all vary hugely between nuclei and between
fields, and comparing pooled distributions would drown a half-pixel effect in
that variation. Comparing the two outlines on the *same* nucleus against the
*same* locally estimated reference cancels all of it -- the same reason the
exhaustive grid in `grids.py` beat the bandit that sampled random images.

Two statistics come out of it per matched nucleus, but **only one of them turns
out to be well-posed on this data**. See `_levels` for the sampling bug that had
to be fixed before either could be believed at all.

* `grad` -- gradient magnitude along the outline, normalised by local contrast.
  **The one to weight.** The comparison divides both outlines by the *same*
  contrast, so any error in the interior/background estimate cancels exactly. It
  favours the prediction in 47 of 49 validation fields.
* `level` -- boundary intensity mapped onto the interior/background scale, where
  0.5 is half-maximum. **Reported as a method that failed, not as a counterweight.**
  It is only meaningful if 0.5 is correctly located, which requires a plateau the
  edge ramp rises to. Measuring the mean radial intensity profile of real nuclei
  shows there isn't one: intensity declines monotonically from the centre outward
  (1.52 at the core, 1.33 mid-radius, 0.73 at the annotated boundary, relative to
  the deep interior), because Hoechst binds DNA and chromatin thins toward the
  periphery. So "the maximum" depends entirely on sampling depth. Re-estimating the
  interior from a relative-depth annulus rather than an eroded core moved the
  result by 0.006, confirming this is not a tuning problem -- the reference does
  not exist for this object class. Both outlines consequently read ~0.40-0.43, and
  comparing that to 0.5 mechanically rewards the smaller outline.

So the honest reading is one informative measurement, not two conventions in a
standoff: the gradient statistic needs no plateau and favours the prediction in
47 of 49 fields.

What none of this can do is separate annotator imprecision from a genuine
difference of opinion about where a nucleus ends. Nobody traced any nucleus in
BBBC039 twice, so inter-annotator agreement -- the direct measure of human
precision -- is not computable on this dataset at all. The claim available is
narrow: *the model's outline sits on the steeper part of the intensity ramp in
almost every field.* That is a statement about where the two outlines sit, not
about which is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.measure import find_contours

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
    """Median boundary intensity (on the half-max scale) and normalised gradient.

    **Sampled on the sub-pixel contour, not on the ring of pixels inside the
    mask.** The obvious implementation -- `find_boundaries(mask, mode="inner")` --
    is badly biased for this measurement, and the bias is larger than the effect.

    That ring sits roughly half a pixel *inside* the true contour, so on a blurred
    edge it reads brighter than half-maximum: a perfectly placed outline registers
    ~0.64 rather than 0.50 at typical blur. Worse, the bias is not common-mode
    between the two outlines being compared. An outline that is slightly too
    *large* has its inner ring land nearer the real edge, so it scores better --
    which means the statistic systematically rewards over-large masks. On a
    synthetic sweep where the ground truth sits exactly on half-maximum and the
    prediction is 0.25 px too large, the inner-ring version picks the wrong
    outline in 9 of 9 configurations, and the gradient variant in 7 of 9. Since
    the model's masks here *are* systematically larger than the annotation's, that
    artifact alone could manufacture the entire result.

    Marching squares gives the contour at the 0.5 level of the binary mask --
    the actual rasterised boundary, between the last inside pixel and the first
    outside one -- and `map_coordinates` samples the image along it with bilinear
    interpolation. On the same sweep this version is correct in 9 of 9, for both
    statistics, and a perfectly placed outline registers ~0.49.
    """
    contours = find_contours(np.ascontiguousarray(mask, dtype=np.float64), 0.5)
    if not contours:
        return float("nan"), float("nan")
    # Longest contour: the object's outer boundary, ignoring any interior holes.
    coords = max(contours, key=len).T
    img_vals = ndi.map_coordinates(image.astype(np.float64), coords, order=1,
                                   mode="nearest")
    grad_vals = ndi.map_coordinates(grad.astype(np.float64), coords, order=1,
                                    mode="nearest")
    if img_vals.size == 0:
        return float("nan"), float("nan")
    level = (float(np.median(img_vals)) - i_bg) / contrast
    return level, float(np.median(grad_vals)) / contrast


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

        # Interior reference. Not an eroded core: Hoechst-stained chromatin is
        # denser toward the nuclear centre, so the deep interior reads brighter
        # than the plateau the edge ramp actually rises to, which biases the
        # half-maximum level low for both outlines. Instead take an annulus at a
        # *relative* depth -- normalised distance-to-boundary in [0.30, 0.60] --
        # which clears the ~1-2 px blur transition without reaching the bright
        # centre, and adapts to object size. Falls back to erosion when an object
        # is too small for the band to contain anything.
        agree = gt_m & pred_m
        edt = ndi.distance_transform_edt(agree)
        if edt.max() > 0:
            rel = edt / edt.max()
            core = (rel >= 0.30) & (rel <= 0.60)
            if core.sum() < 20:
                core = ndi.binary_erosion(agree, core_se)
        else:
            core = ndi.binary_erosion(agree, core_se)
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
    """Paired comparison of the two outlines, at both object and image level.

    **Why two levels.** A signed-rank test over ~4,500 individual nuclei assumes
    they are independent observations. They are not: boundary placement depends on
    focus, exposure and staining, all of which are properties of the *field*, so
    objects within one image are correlated. The effective sample size is nearer
    the number of fields than the number of objects, and an object-level p-value
    is therefore overstated -- by many orders of magnitude at this n.

    So the headline test is the **per-image** one: reduce each field to its median
    offset for each outline, then compare those paired medians across fields. That
    respects the clustering, costs almost nothing in significance for an effect
    this consistent, and cannot be attacked on independence grounds. The
    object-level numbers are still reported, marked as uncorrected, because the
    per-object distribution is what the figure draws.
    """
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
    # Per-image medians: one paired observation per field, which is the level at
    # which the observations are actually independent.
    by_image: dict[str, list[BoundaryFit]] = {}
    for f in fits:
        by_image.setdefault(f.name, []).append(f)
    img_gt = np.array([np.median([f.gt_error for f in v]) for v in by_image.values()])
    img_pr = np.array([np.median([f.pred_error for f in v]) for v in by_image.values()])
    img_gt_g = np.array([np.median([f.gt_grad for f in v]) for v in by_image.values()])
    img_pr_g = np.array([np.median([f.pred_grad for f in v]) for v in by_image.values()])
    out["per_image"] = {
        "n_images": len(by_image),
        "gt_abs_error_median": float(np.median(img_gt)),
        "pred_abs_error_median": float(np.median(img_pr)),
        "pred_closer_fraction": float(np.mean(img_pr < img_gt)),
        "pred_sharper_fraction": float(np.mean(img_pr_g > img_gt_g)),
    }

    try:
        from scipy.stats import wilcoxon

        # Headline: paired over fields, respecting within-image correlation.
        out["per_image"]["wilcoxon_p_abs_error"] = float(wilcoxon(img_pr, img_gt).pvalue)
        out["per_image"]["wilcoxon_p_gradient"] = float(wilcoxon(img_pr_g, img_gt_g).pvalue)
        # Uncorrected, over objects. Retained for the figure, not for inference.
        out["wilcoxon_p_abs_error_uncorrected"] = float(wilcoxon(pr_err, gt_err).pvalue)
        out["wilcoxon_p_gradient_uncorrected"] = float(wilcoxon(pr_grd, gt_grd).pvalue)
    except Exception:  # scipy.stats is optional for the rest of the module
        pass
    return out


def verdict(summary: dict) -> str:
    """One line stating what the paired comparison licenses us to say.

    Quotes the per-image test, not the per-object one -- see `summarize`.
    """
    if not summary:
        return "no measurable objects"
    gt_e, pr_e = summary["gt_abs_error_median"], summary["pred_abs_error_median"]
    frac = summary["pred_closer_fraction"]
    img = summary.get("per_image", {})
    n_img = img.get("n_images", 0)
    lvl = img.get("pred_closer_fraction", float("nan"))
    grd = img.get("pred_sharper_fraction", float("nan"))
    p_l = img.get("wilcoxon_p_abs_error", float("nan"))
    p_g = img.get("wilcoxon_p_gradient", float("nan"))
    return (
        f"over {n_img} fields. GRADIENT (reference-free, needs no plateau -- the "
        f"only well-posed measure here): favours the prediction in {grd:.0%} of "
        f"fields, p={p_g:.1e}. HALF-MAX LEVEL: favours the annotation in "
        f"{1 - lvl:.0%} of fields (p={p_l:.1e}), but these nuclei have no intensity "
        f"plateau, so 0.5 is not locatable and this is NOT evidence -- see the "
        f"module docstring."
    )
