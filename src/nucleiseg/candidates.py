"""Features for deciding whether a residual-pass candidate is a real nucleus.

`smallobj.py` proposes small objects the first pass missed, at about 28%
precision. That is a *proposal* stage, and its ceiling was reached by hand-tuning
three thresholds — sweeping them further loses recall without gaining precision,
because a single threshold on any one cue cannot separate the classes.

This module supplies the second stage: describe each candidate with features that
a model can weigh *jointly*, and let it learn the boundary. The training signal is
free — the ground truth says which candidates correspond to real missed nuclei.

**Why hand-designed features rather than a small CNN on patches.** There are only
a few hundred positives available in the training split, which is far too few to
train a convolutional model that would not simply memorise. Ten interpretable
features with a gradient-boosted tree is the right capacity for this sample size,
and it has the side benefit that the fitted model can be interrogated: if
signal-to-noise dominates, that is a statement about the imaging, not just about
the classifier.

**The features and why each is here.** Every one is computed from the image and
the first-pass prediction only — never from ground truth — so the same code runs
unchanged at inference.

* `area`, `eccentricity`, `solidity` — shape. Nuclei are round and convex; noise
  spikes and debris often are not.
* `peak_contrast`, `mean_contrast` — brightness above local background, in units
  of the field's own object contrast, so it is exposure-invariant.
* `snr` — peak contrast divided by the local background standard deviation. This
  is the one that should matter most if the objects are noise-limited, which is
  what every previous experiment implied.
* `edge_ratio` — boundary gradient energy over interior gradient energy. A real
  object has a coherent rim; a noise spike has gradient everywhere.
* `edge_coverage` — what fraction of the boundary actually carries above-median
  gradient. Distinguishes a closed rim from one bright arc.
* `log_response` — the scale-selective detector's own confidence.
* `dist_to_object` — distance to the nearest first-pass detection. Debris tends to
  sit away from cells; a genuinely missed nucleus often sits among them.
* `local_density` — how many first-pass objects are nearby, as crowding context.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.measure import regionprops

FEATURES = [
    "area", "eccentricity", "solidity",
    "peak_contrast", "mean_contrast", "snr",
    "edge_ratio", "edge_coverage",
    "log_response", "dist_to_object", "local_density",
]


def describe(
    image: np.ndarray,
    first_pass: np.ndarray,
    candidate: np.ndarray,
    log_response: float = 0.0,
) -> dict:
    """Feature vector for one candidate mask. Uses no ground truth."""
    img = image.astype(np.float32)
    found = first_pass > 0
    bg_global = float(np.median(img[~found])) if (~found).any() else float(np.median(img))
    contrast = (
        max(float(np.median(img[found])) - bg_global, 1.0) if found.any() else 1.0
    )

    ys, xs = np.nonzero(candidate)
    if len(ys) == 0:
        return {k: 0.0 for k in FEATURES}
    pad = 8
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    win, m = img[y0:y1, x0:x1], candidate[y0:y1, x0:x1]
    ring_bg = ndi.binary_dilation(m, iterations=4) & ~ndi.binary_dilation(m, iterations=1)
    local_bg = float(np.median(win[ring_bg])) if ring_bg.any() else bg_global
    local_sd = float(np.std(win[ring_bg])) if ring_bg.sum() > 4 else 1.0

    peak = float(win[m].max()) - local_bg
    mean = float(win[m].mean()) - local_bg

    grad = sobel(ndi.gaussian_filter(win, 0.8))
    rim = ndi.binary_dilation(m, iterations=1) & ~ndi.binary_erosion(m)
    interior = ndi.binary_erosion(m)
    e_rim = float(np.mean(grad[rim])) if rim.any() else 0.0
    e_in = float(np.mean(grad[interior])) if interior.any() else float(np.mean(grad[m]))
    # What fraction of the rim actually carries edge energy, vs one bright arc.
    coverage = float(np.mean(grad[rim] > np.median(grad))) if rim.any() else 0.0

    props = regionprops(m.astype(np.uint8))
    prop = props[0] if props else None

    # Context: how far to the nearest real detection, and how crowded it is here.
    if found.any():
        dist = float(ndi.distance_transform_edt(~found)[ys[0], xs[0]])
        yy0, yy1 = max(ys.min() - 60, 0), min(ys.max() + 60, first_pass.shape[0])
        xx0, xx1 = max(xs.min() - 60, 0), min(xs.max() + 60, first_pass.shape[1])
        density = float(len(np.unique(first_pass[yy0:yy1, xx0:xx1])) - 1)
    else:
        dist, density = 0.0, 0.0

    return {
        "area": float(candidate.sum()),
        "eccentricity": float(prop.eccentricity) if prop else 0.0,
        "solidity": float(prop.solidity) if prop else 0.0,
        "peak_contrast": peak / contrast,
        "mean_contrast": mean / contrast,
        "snr": peak / max(local_sd, 1e-6),
        "edge_ratio": e_rim / max(e_in, 1e-6),
        "edge_coverage": coverage,
        "log_response": float(log_response),
        "dist_to_object": dist,
        "local_density": density,
    }


def to_matrix(rows: list[dict]) -> np.ndarray:
    """Feature dicts -> array in the fixed FEATURES order."""
    return np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float64)
