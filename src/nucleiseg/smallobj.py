"""A second pass over the residual, to recover the small faint objects.

**Why a residual pass rather than another global knob.** Every global intervention
tried so far failed for the same reason: lowering a threshold to admit small faint
objects admits noise *everywhere*, and the noise outweighs the objects. The
`min_size` sweep, the `cellprob_threshold` sweep and local normalisation all lose
on that trade.

But the trade is only forced if the second look is global. After the first pass,
the ~5,600 confidently detected nuclei can be erased from the image, and what
remains is background plus the ~350 things the model missed. **In that residual the
missed objects are the brightest structures present**, so they can be found with a
locally adaptive detector that would be hopeless on the original field. The first
pass is not being re-tuned; it is being used to define where the second pass is
allowed to look.

**Why Laplacian-of-Gaussian for the detector.** It is scale-selective by
construction -- the response peaks when the filter scale matches the blob radius --
which is exactly the prior available here: the missed population is 3-10 px across
and roughly round. A plain threshold has no scale prior and would fire on any
bright pixel.

**Why a high-frequency gate on the candidates.** Adapted from the HFEF idea of
using high-frequency energy as an input-level cue (Sahoo et al., stationary wavelet
+ Laplacian energy). A real nucleus, however faint, has a *coherent boundary*: high
gradient energy arranged in a closed ring around a brighter interior. A shot-noise
spike does not. Scoring candidates on the ratio of edge-ring energy to interior
energy separates the two without needing a trained model, and it is the one place
where a global brightness threshold genuinely cannot substitute.

Nothing here is trained. Every parameter is either measured from the failure
analysis (the size range) or fixed by the geometry of the detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log
from skimage.filters import sobel


@dataclass(frozen=True)
class ResidualParams:
    """All defaults come from the measured miss population, not from tuning."""

    # Missed objects: median 20 px area, 81% under 100 px. Sigma range covers
    # radii of roughly 1.5-5 px, i.e. 3-10 px across.
    min_sigma: float = 1.0
    max_sigma: float = 3.5
    num_sigma: int = 6
    # LoG response floor, on the contrast-normalised image.
    threshold: float = 0.12
    # How far to dilate first-pass objects before erasing them, so their blurred
    # edges do not survive as candidates.
    erase_dilation: int = 2
    # A candidate must clear the local background by this fraction of the field's
    # own object contrast -- scale-free, so it adapts to exposure.
    min_rel_contrast: float = 0.35
    # Edge-ring energy must exceed interior energy by this factor (the HF gate).
    min_edge_ratio: float = 1.35
    # Where to cut the local intensity ramp when turning a detection into a mask.
    # 0.5 (half-height) is the textbook choice and is badly wrong here: on faint
    # objects it yields masks ~41% of true area, so a correctly *located* candidate
    # still fails IoU 0.5. Swept on validation, hit rate roughly triples from 13%
    # to 40% as this drops to 0.20, then plateaus.
    fill_fraction: float = 0.20
    # Accept only objects in the size band the first pass actually misses.
    min_area: int = 4
    max_area: int = 120


def _residual(image: np.ndarray, labels: np.ndarray, dilation: int) -> np.ndarray:
    """The image with everything the first pass found removed."""
    found = labels > 0
    if dilation > 0:
        found = ndi.binary_dilation(found, ndi.generate_binary_structure(2, 2),
                                    iterations=dilation)
    bg = float(np.median(image[~found])) if (~found).any() else 0.0
    out = image.astype(np.float32).copy()
    out[found] = bg
    return out


def _edge_ratio(image: np.ndarray, mask: np.ndarray) -> float:
    """Boundary gradient energy relative to interior gradient energy.

    A real object has a coherent edge and a smooth interior, so this is well
    above 1. A noise spike has gradient energy everywhere and sits near 1.
    """
    if mask.sum() < 3:
        return 0.0
    grad = sobel(ndi.gaussian_filter(image, 0.8))
    ring = ndi.binary_dilation(mask, iterations=1) & ~ndi.binary_erosion(mask)
    interior = ndi.binary_erosion(mask)
    if not ring.any():
        return 0.0
    e_ring = float(np.mean(grad[ring]))
    e_in = float(np.mean(grad[interior])) if interior.any() else float(np.mean(grad[mask]))
    return e_ring / max(e_in, 1e-6)


def propose(
    image: np.ndarray,
    labels: np.ndarray,
    params: ResidualParams | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Candidate masks and the detector's own response for each.

    Split out from `find_missed` so a downstream classifier can score candidates
    on more than the three hand-set gates. The gates here are deliberately kept
    loose when `params.threshold` is lowered: recall at the proposal stage is what
    bounds the whole two-stage system, and precision is the classifier's job.
    """
    p = params or ResidualParams()
    out = []
    for mask, resp in _propose_raw(image, labels, p):
        out.append((mask, resp))
    return out


def _propose_raw(image, labels, p):
    """Shared candidate generation. Yields (full-size boolean mask, LoG response)."""
    resid = _residual(image, labels, p.erase_dilation)
    found = labels > 0
    bg = float(np.median(resid))
    if found.any():
        contrast = max(float(np.median(image[found])) - bg, 1.0)
    else:
        contrast = max(float(resid.max()) - bg, 1.0)
    norm = np.clip((resid - bg) / contrast, 0, None)
    blobs = blob_log(norm, min_sigma=p.min_sigma, max_sigma=p.max_sigma,
                     num_sigma=p.num_sigma, threshold=p.threshold, overlap=0.5)
    h, w = labels.shape
    claimed = np.zeros(labels.shape, dtype=bool)
    for y, x, sigma in blobs:
        y, x = int(round(y)), int(round(x))
        r = int(max(4, round(sigma * 3)))
        y0, y1 = max(y - r, 0), min(y + r + 1, h)
        x0, x1 = max(x - r, 0), min(x + r + 1, w)
        win = norm[y0:y1, x0:x1]
        if win.size == 0:
            continue
        peak = float(win.max())
        local_bg = float(np.median(win))
        if peak - local_bg < p.min_rel_contrast:
            continue
        cc, n = ndi.label(win >= local_bg + p.fill_fraction * (peak - local_bg))
        if n == 0:
            continue
        pid = cc[np.unravel_index(np.argmax(win), win.shape)]
        if pid == 0:
            continue
        blob_mask = cc == pid
        area = int(blob_mask.sum())
        if not (p.min_area <= area <= p.max_area):
            continue
        if (labels[y0:y1, x0:x1][blob_mask] > 0).any():
            continue
        if (claimed[y0:y1, x0:x1][blob_mask]).any():
            continue
        if _edge_ratio(win, blob_mask) < p.min_edge_ratio:
            continue
        full = np.zeros(labels.shape, dtype=bool)
        full[y0:y1, x0:x1][blob_mask] = True
        claimed |= full
        yield full, float(peak - local_bg)


def find_missed(
    image: np.ndarray,
    labels: np.ndarray,
    params: ResidualParams | None = None,
    keep: "callable | None" = None,
) -> np.ndarray:
    """Return a label image of newly found small objects (0 where nothing added).

    `keep(mask, response) -> bool` optionally filters candidates; this is where a
    trained classifier plugs in, replacing the hand-set gates.
    """
    p = params or ResidualParams()
    out = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for mask, resp in _propose_raw(image, labels, p):
        if keep is not None and not keep(mask, resp):
            continue
        out[mask] = next_id
        next_id += 1
    return out


def augment_labels(
    image: np.ndarray, labels: np.ndarray, params: ResidualParams | None = None,
    keep: "callable | None" = None,
) -> tuple[np.ndarray, int]:
    """First-pass labels plus recovered small objects, renumbered. Returns (labels, n_added)."""
    extra = find_missed(image, labels, params, keep=keep)
    n_added = int(extra.max())
    if n_added == 0:
        return labels.copy(), 0
    merged = labels.copy().astype(np.int32)
    offset = int(labels.max())
    merged[extra > 0] = extra[extra > 0] + offset
    return merged, n_added
