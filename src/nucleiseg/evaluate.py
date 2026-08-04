"""Evaluation harness: run a segmenter over a split and report metrics.

STUB. Intended CLI:

    python -m nucleiseg.evaluate --split validation
    python -m nucleiseg.evaluate --split test --params results/best_params.json

Requirements it needs to satisfy:

* Write per-image rows (name, n_gt, n_pred, AP, F1@0.5, splits, merges) to
  results/<tag>_per_image.csv, not just a single aggregate. The failure
  analysis is driven by finding the worst images and looking at them, which is
  impossible from a mean.
* Aggregate by *mean over images*, and say so. Pooling all objects across
  images instead weights dense fields more heavily and hides the empty-field
  failure entirely.
* Handle the degenerate cases explicitly rather than dividing by zero: three
  fields in this dataset have zero GT nuclei (two in training, one in
  validation), so a field can be a perfect score with no objects or an
  all-false-positive score. Decide and document how those fields enter the
  mean.
* Be deterministic and record the parameters used alongside the numbers, so a
  before/after comparison is actually reproducible.
"""

from __future__ import annotations


def evaluate_split(split: str, tag: str = "baseline") -> dict:
    """Run the segmenter over a split; return aggregates and write per-image CSV."""
    raise NotImplementedError


def sweep(split: str, param_grid: dict) -> None:
    """Grid/coordinate search over BaselineParams on a split, logging every point.

    Tune on validation only. The single deliberate optimization for the writeup
    comes out of here, so it must save the full sweep table (not only the
    winner) to show the shape of the response curve.
    """
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
