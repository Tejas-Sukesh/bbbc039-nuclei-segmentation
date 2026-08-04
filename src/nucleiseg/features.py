"""Cheap per-image descriptors used as context for the contextual bandit.

Two hard constraints shape this module.

**No ground truth.** Every feature must be computable from the raw image alone.
Using labels to choose per-image parameters at evaluation time would be leakage
dressed up as a method: the policy has to map image appearance to a parameter
choice and then run blind. This is the first thing a careful reader will check.

**Very few features.** With ~100 training images, a rich context vector will
memorise rather than generalise. Three features plus a bias term is deliberately
close to the smallest thing that could work, and the point is to stay on the
right side of that tradeoff rather than to maximise expressiveness.

The three chosen each target a failure mode that was actually observed:

* `foreground_fraction` -- proxy for confluency. Merge errors scale with
  crowding, so this is the feature most likely to matter.
* `sharpness` -- variance of the Laplacian, the standard focus measure. Defocus
  flattens edge gradients and shifts where a threshold lands, so out-of-focus
  fields want different parameters.
* `bright_tail` -- how heavy the high-intensity tail is, which is the signature
  of debris or saturation skewing a global threshold.

Features are standardised using statistics fitted on the training split only, and
those statistics are saved with the policy: recomputing them on the evaluation
split would let test-set information influence the encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

FEATURE_NAMES = ("foreground_fraction", "sharpness", "bright_tail")


def raw_features(image: np.ndarray) -> np.ndarray:
    """Three unstandardised descriptors of a raw uint16 field."""
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, [1.0, 99.8])
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(img)

    # Confluency proxy. Otsu can fail on a near-empty field, hence the guard.
    try:
        fg = float((norm > threshold_otsu(norm)).mean())
    except ValueError:
        fg = 0.0

    # Focus. Laplacian variance on the normalised image, so it is comparable
    # across fields with different gain.
    sharpness = float(ndi.laplace(norm).var())

    # Heaviness of the bright tail, relative to the bulk of the signal.
    p50, p999 = np.percentile(img, [50.0, 99.9])
    spread = max(hi - lo, 1e-6)
    bright_tail = float((p999 - p50) / spread)

    return np.array([fg, sharpness, bright_tail], dtype=np.float64)


@dataclass
class FeatureScaler:
    """Standardises features to zero mean / unit variance and appends a bias term.

    Fit on the training split only. `transform` returns a vector of length
    ``len(FEATURE_NAMES) + 1``; the trailing 1.0 lets each arm's linear model
    learn an intercept.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feats: np.ndarray) -> "FeatureScaler":
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float64))
        std = feats.std(axis=0)
        std[std < 1e-9] = 1.0  # a constant feature carries no information
        return cls(mean=feats.mean(axis=0), std=std)

    def transform(self, feats: np.ndarray) -> np.ndarray:
        f = np.asarray(feats, dtype=np.float64)
        z = (f - self.mean) / self.std
        if z.ndim == 1:
            return np.concatenate([z, [1.0]])
        return np.hstack([z, np.ones((z.shape[0], 1))])

    @property
    def n_features(self) -> int:
        return len(self.mean) + 1

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(mean=np.array(d["mean"]), std=np.array(d["std"]))
