"""Classical segmentation baseline: normalize -> threshold -> seed -> watershed.

Kept in the project as the interpretable reference. Every stage is inspectable,
so when a field scores badly the error can be attributed to a *stage* rather
than to a black box -- which is what makes a mechanism-level failure analysis
possible at all.

The stage that matters is **seeding**. Thresholding fuses touching nuclei into
one connected component; the distance transform of that fused mask still has one
local maximum per nucleus with a saddle at the neck between them, so watershed
can recover the separation from geometry alone. Whether it does comes down to
how prominent those maxima are, which is what `h_maxima` controls. Splits and
merges are both *seeding* failures, not watershed failures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, threshold_otsu, threshold_triangle
from skimage.morphology import h_maxima, remove_small_objects
from skimage.segmentation import watershed


@dataclass(frozen=True)
class BaselineParams:
    """Every knob in one place, so a sweep is a loop over dataclass instances."""

    p_low: float = 1.0  # low percentile for intensity rescaling
    p_high: float = 99.8  # high percentile
    threshold: str = "otsu"  # 'otsu' | 'local' | 'triangle'
    local_block: int = 151  # window for adaptive thresholding (odd)
    min_distance: int = 7  # min separation between watershed seeds
    h_maxima: float = 0.0  # prominence floor for seeds, in EDT pixels (0 = off)
    min_area: int = 10  # discard predicted objects smaller than this
    fill_holes: bool = True

    def replace(self, **kw) -> "BaselineParams":
        return replace(self, **kw)


def normalize(image: np.ndarray, params: BaselineParams) -> np.ndarray:
    """Per-image percentile rescale to [0, 1] float32.

    Per-image rather than global because gain and offset vary field to field;
    percentiles rather than min/max because a single saturated debris pixel would
    otherwise set the scale. Note this dataset's floor is ~120, not 0 -- the
    camera black level -- so assuming 0 would compress all real signal into the
    top few percent of the range.
    """
    lo, hi = np.percentile(image, [params.p_low, params.p_high])
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def foreground_mask(norm: np.ndarray, params: BaselineParams) -> np.ndarray:
    """Binary nuclei-vs-background mask.

    Otsu maximizes between-class variance under a two-class model, so it assumes
    a bimodal histogram. That assumption is the documented weak point: on a
    near-empty field the histogram is unimodal sensor noise and Otsu still
    returns a threshold, manufacturing nuclei out of noise.
    """
    if params.threshold == "otsu":
        mask = norm > threshold_otsu(norm)
    elif params.threshold == "triangle":
        mask = norm > threshold_triangle(norm)
    elif params.threshold == "local":
        block = params.local_block | 1  # must be odd
        mask = norm > threshold_local(norm, block_size=block)
    else:
        raise ValueError(f"unknown threshold {params.threshold!r}")
    if params.fill_holes:
        mask = ndi.binary_fill_holes(mask)
    return mask


def find_seeds(mask: np.ndarray, params: BaselineParams) -> tuple[np.ndarray, np.ndarray]:
    """One marker per nucleus from the distance transform. Returns (markers, edt).

    The EDT gives each foreground pixel its distance to the nearest background
    pixel -- the radius of the largest circle centred there that fits inside the
    mask -- so a local maximum sits at a locally most-interior point. For a
    roughly convex, roughly round nucleus that is one point at the centre.
    """
    edt = ndi.distance_transform_edt(mask)
    if params.h_maxima > 0:
        # Prominence filter: drop maxima less than h above the saddle connecting
        # them to a taller maximum. Scale-free, unlike a fixed-radius NMS.
        peaks = h_maxima(edt, params.h_maxima) > 0
    else:
        peaks = np.zeros(mask.shape, dtype=bool)
        coords = peak_local_max(
            edt, min_distance=params.min_distance, labels=mask, exclude_border=False
        )
        if coords.size:
            peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(peaks)
    return markers, edt


def segment(image: np.ndarray, params: BaselineParams | None = None) -> np.ndarray:
    """Full pipeline: raw uint16 image -> int32 instance labels (0 = background)."""
    params = params or BaselineParams()
    norm = normalize(image, params)
    mask = foreground_mask(norm, params)
    if not mask.any():
        return np.zeros(image.shape, dtype=np.int32)

    markers, edt = find_seeds(mask, params)
    if markers.max() == 0:
        # No seed survived; fall back to connected components so a merged blob
        # is reported rather than nothing at all.
        labels, _ = ndi.label(mask)
    else:
        # Flood the *negated* EDT: hills become basins, and basins meet at the
        # saddle between two nuclei. This is a shape argument, not an intensity
        # one -- the image is not used here at all.
        labels = watershed(-edt, markers=markers, mask=mask)

    if params.min_area > 1:
        keep = remove_small_objects(labels > 0, min_size=params.min_area)
        labels = np.where(keep, labels, 0)
    return labels.astype(np.int32)
