"""Figures for the writeup.

STUB. What the failure analysis actually needs (and nothing more):

* `overlay` — raw image with GT boundaries in one color and predicted
  boundaries in another. Boundaries, not filled masks: filled overlays hide
  exactly the boundary disagreements that drive AP at high IoU thresholds.
* `error_panel` — for one image, a 4-up of raw / GT / prediction / error map,
  where the error map color-codes true positives, false negatives, false
  positives, splits, and merges. This is the figure that makes a failure mode
  legible in the README.
* `worst_cases` — given the per-image CSV from `evaluate`, render error panels
  for the N lowest-scoring fields. The failure writeup should be built from
  these rather than from cherry-picked examples.
* `stage_panel` — the intermediate products of the classical pipeline
  (normalized, foreground mask, distance transform, seeds, watershed) for a
  single image, to attribute an error to a stage.
* `sweep_curve` — metric vs. the swept parameter, for the before/after section.

Use a random-permutation colormap for label images; a sequential colormap makes
adjacent instances nearly indistinguishable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out: Path) -> None:
    raise NotImplementedError


def error_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out: Path) -> None:
    raise NotImplementedError


def worst_cases(per_image_csv: Path, n: int, out_dir: Path) -> None:
    raise NotImplementedError


def stage_panel(image: np.ndarray, out: Path) -> None:
    raise NotImplementedError


def sweep_curve(sweep_csv: Path, param: str, out: Path) -> None:
    raise NotImplementedError
