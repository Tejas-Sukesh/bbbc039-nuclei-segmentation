"""Segmenters behind one interface, plus the flow cache that makes tuning cheap.

Every segmenter maps a raw uint16 image to an int32 label image (0 = background),
matching the convention of `data.decode_mask`, so the metrics module can compare
any of them against ground truth without special cases.

The important engineering decision lives in `FlowCache`. Cellpose-SAM inference
costs ~9 s per field on Apple Silicon MPS, which would make a parameter search
impractical: a few hundred trials over 50 images is hours of compute. But the
expensive part is only the network forward pass, which produces a flow field
(`dP`) and a cell-probability map (`cellprob`). Turning those into instance
labels is pure post-processing and takes ~0.44 s.

So the network runs **once per image**, its output is cached to disk, and the
tunable post-processing parameters are then swept over the cache at roughly 20x
the speed. Verified bit-identical to calling `model.eval()` directly.

That split also partitions the parameters, which is what the optimizer needs to
know:

* **cheap** (recomputed from cache): `cellprob_threshold`, `flow_threshold`,
  `min_size`, `max_size_fraction`, `niter`
* **expensive** (require re-running the network): `augment`, `normalize`,
  `diameter`, tiling

Conveniently, both mechanisms behind the measured -5.1% count bias --
`min_size=15` discarding real nuclei, and `cellprob_threshold=0.0` -- are cheap
parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from .data import REPO_ROOT, load_sample

CACHE_DIR = REPO_ROOT / "data" / "flow_cache"


class Segmenter(Protocol):
    """Anything that turns a raw image into instance labels."""

    name: str

    def segment(self, image: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Cellpose-SAM
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellposeParams:
    """Post-processing parameters recomputable from cached flows.

    Defaults are Cellpose's own, so `CellposeParams()` reproduces out-of-the-box
    behaviour exactly and is the honest "before" in any before/after comparison.
    """

    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    min_size: int = 15
    max_size_fraction: float = 0.4
    niter: int = 200

    def replace(self, **kw) -> "CellposeParams":
        return replace(self, **kw)

    def key(self) -> tuple:
        return tuple(sorted(asdict(self).items()))


def _resolve_device(device: str | None):
    import torch

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


class FlowCache:
    """Runs the Cellpose network once per image and caches (dP, cellprob) to disk.

    Cached arrays are stored compressed as float16 -- they are intermediate
    activations, not measurements, and halving the footprint costs nothing
    measurable in the resulting masks.
    """

    def __init__(
        self,
        model_name: str = "cpsam_v2",
        device: str | None = None,
        cache_dir: Path | None = None,
        augment: bool = False,
    ):
        self.model_name = model_name
        self.augment = augment
        self.device = _resolve_device(device)
        # Separate subdirectory per network-level config, since those parameters
        # change the cached flows themselves.
        tag = f"{model_name}{'_aug' if augment else ''}"
        self.cache_dir = (cache_dir or CACHE_DIR) / tag
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from cellpose import models

            self._model = models.CellposeModel(
                gpu=self.device.type != "cpu",
                pretrained_model=self.model_name,
                device=self.device,
            )
        return self._model

    def path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.npz"

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def build(self, name: str, image: np.ndarray | None = None) -> None:
        """Run the network on one image and cache its flows."""
        if self.has(name):
            return
        if image is None:
            image = load_sample(name).image
        _, flows, _ = self.model.eval(image, augment=self.augment)
        dP, cellprob = flows[1], flows[2]
        np.savez_compressed(
            self.path(name),
            dP=dP.astype(np.float16),
            cellprob=cellprob.astype(np.float16),
        )

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.has(name):
            self.build(name)
        z = np.load(self.path(name))
        return z["dP"].astype(np.float32), z["cellprob"].astype(np.float32)

    def masks(self, name: str, params: CellposeParams) -> np.ndarray:
        """Recompute instance labels from cached flows -- the cheap path."""
        from cellpose import dynamics

        dP, cellprob = self.load(name)
        out = dynamics.resize_and_compute_masks(
            dP,
            cellprob,
            niter=params.niter,
            cellprob_threshold=params.cellprob_threshold,
            flow_threshold=params.flow_threshold,
            min_size=params.min_size,
            max_size_fraction=params.max_size_fraction,
            device=self.device,
        )
        masks = out[0] if isinstance(out, tuple) else out
        return np.asarray(masks, dtype=np.int32)


class CellposeSegmenter:
    """Cellpose-SAM as a `Segmenter`, going through the cache when given a name."""

    def __init__(
        self,
        params: CellposeParams | None = None,
        cache: FlowCache | None = None,
        model_name: str = "cpsam_v2",
        device: str | None = None,
    ):
        self.params = params or CellposeParams()
        self.cache = cache or FlowCache(model_name=model_name, device=device)
        self.name = f"cellpose:{model_name}"

    def segment_named(self, name: str) -> np.ndarray:
        return self.cache.masks(name, self.params)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Uncached path, for an image with no dataset identity."""
        from cellpose import dynamics

        _, flows, _ = self.cache.model.eval(image, augment=self.cache.augment)
        out = dynamics.resize_and_compute_masks(
            flows[1],
            flows[2],
            niter=self.params.niter,
            cellprob_threshold=self.params.cellprob_threshold,
            flow_threshold=self.params.flow_threshold,
            min_size=self.params.min_size,
            max_size_fraction=self.params.max_size_fraction,
            device=self.cache.device,
        )
        masks = out[0] if isinstance(out, tuple) else out
        return np.asarray(masks, dtype=np.int32)


# --------------------------------------------------------------------------- #
# Classical
# --------------------------------------------------------------------------- #


class ClassicalSegmenter:
    """The threshold -> distance transform -> watershed pipeline as a `Segmenter`."""

    def __init__(self, params=None):
        from .baseline import BaselineParams

        self.params = params or BaselineParams()
        self.name = "classical"

    def segment(self, image: np.ndarray) -> np.ndarray:
        from .baseline import segment

        return segment(image, self.params)
