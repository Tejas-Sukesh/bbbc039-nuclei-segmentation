"""Classical segmentation baseline: threshold -> seed -> watershed.

STUB. The plan, and why it is this rather than a network:

The point of a classical baseline here is that every stage is inspectable, so
when the metric is bad we can say *which stage* lost the nuclei. A U-Net that
scores 0.05 higher tells us much less about the failure modes, which is what
this exercise actually asks for.

Pipeline stages, each an explicit knob:

1. Normalize. Raw images are uint16 but only occupy ~120..4095 (the camera
   floor is ~120, not 0). Per-image percentile rescaling (e.g. 1st..99.9th)
   rather than a global constant, because plate-to-plate illumination varies.
2. Foreground. Global Otsu is the obvious start and the documented weak point
   -- it assumes a bimodal histogram, which breaks on fields that are nearly
   empty (three fields in this dataset have zero nuclei) or that carry a bright
   debris blob dragging the threshold up. Compare against local/adaptive
   thresholding.
3. Seeds. Euclidean distance transform of the foreground, then local maxima as
   one seed per nucleus. This is the stage that decides whether touching nuclei
   get split, and `h` (h-maxima suppression depth) / min-distance is the single
   most consequential parameter in the pipeline -- the natural candidate for
   the "optimize one thing deliberately" requirement.
4. Watershed. Flood from the seeds, constrained to the foreground mask.
5. Post-filter. Drop objects below a minimum area (GT nuclei run ~13 px at the
   smallest, so this threshold must stay low), optionally fill holes.

All parameters are tuned on `validation` only and reported once on `test`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineParams:
    """Every knob in one place, so a sweep is a loop over dataclass instances."""

    p_low: float = 1.0  # low percentile for intensity rescaling
    p_high: float = 99.9  # high percentile
    threshold: str = "otsu"  # 'otsu' | 'local' | 'triangle'
    local_block: int = 151  # window for adaptive thresholding
    min_distance: int = 7  # min separation between watershed seeds
    h_maxima: float = 0.0  # h-maxima suppression depth (0 = off)
    min_area: int = 20  # discard predicted objects smaller than this
    fill_holes: bool = True


def normalize(image: np.ndarray, params: BaselineParams) -> np.ndarray:
    """Per-image percentile rescale to [0, 1] float32."""
    raise NotImplementedError


def foreground_mask(norm: np.ndarray, params: BaselineParams) -> np.ndarray:
    """Binary nuclei-vs-background mask."""
    raise NotImplementedError


def find_seeds(mask: np.ndarray, params: BaselineParams) -> np.ndarray:
    """One marker per nucleus, as an int label image, from the distance transform."""
    raise NotImplementedError


def segment(image: np.ndarray, params: BaselineParams | None = None) -> np.ndarray:
    """Full pipeline: raw uint16 image -> int32 instance labels (0 = background).

    Output must match the convention of `data.decode_mask` so the metrics
    module can compare them directly.
    """
    raise NotImplementedError
