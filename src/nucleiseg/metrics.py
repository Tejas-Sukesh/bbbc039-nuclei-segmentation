"""Instance-segmentation metrics against ground-truth labels.

STUB — signatures and the intended semantics are fixed here so the evaluation
contract is settled before any segmenter is written. See RESOURCES.md §2 for
the references behind these choices.

Design decisions already made (do not silently change them):

* Match predicted to GT instances by IoU, one-to-one. Greedy matching by
  descending IoU is the Data Science Bowl 2018 convention and is what the
  Caicedo et al. evaluation uses; it can differ from optimal (Hungarian)
  assignment in dense clumps, so `match_instances` takes a `strategy` flag and
  the writeup should report whether the two disagree.
* Report average precision over IoU thresholds 0.50:0.05:0.95 (the DSB2018
  metric), *and* F1 at a single IoU of 0.5. The sweep is the headline number
  because it is sensitive to boundary quality; F1@0.5 is the number that is
  legible to a biologist counting cells.
* Also track split and merge counts separately. A pipeline that over-segments
  and one that under-segments can post identical AP while failing in opposite
  directions, and the failure writeup needs to tell them apart.
* Instance IDs are assumed contiguous 1..N with 0 = background, as produced by
  `data.decode_mask`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pairwise IoU between every GT and predicted instance.

    Returns an (n_gt, n_pred) array. Implement via a 2-D histogram of
    (gt_label, pred_label) pixel pairs to get all intersections in one pass --
    looping over instance pairs is O(N*M) mask comparisons and is far too slow
    at ~120 nuclei per image over 50 images.
    """
    raise NotImplementedError


@dataclass
class MatchResult:
    """One-to-one assignment between GT and predicted instances at one IoU threshold."""

    threshold: float
    matched: list[tuple[int, int]]  # (gt_id, pred_id) pairs above threshold
    false_negatives: list[int]  # unmatched GT ids (missed nuclei)
    false_positives: list[int]  # unmatched pred ids (spurious objects)

    @property
    def precision(self) -> float:
        raise NotImplementedError

    @property
    def recall(self) -> float:
        raise NotImplementedError

    @property
    def f1(self) -> float:
        raise NotImplementedError


def match_instances(
    ious: np.ndarray, threshold: float, strategy: str = "greedy"
) -> MatchResult:
    """One-to-one match of GT to predictions at an IoU threshold.

    `strategy` is 'greedy' (DSB2018 convention) or 'hungarian' (optimal, via
    scipy.optimize.linear_sum_assignment).
    """
    raise NotImplementedError


def average_precision(gt: np.ndarray, pred: np.ndarray) -> dict[float, float]:
    """DSB2018-style AP per IoU threshold.

    At each threshold, AP = TP / (TP + FP + FN) -- note this is the Kaggle
    definition, which is a Jaccard-style ratio over objects, NOT the
    area-under-the-precision-recall-curve AP from object detection. They are
    different numbers; the writeup must say which one it reports.
    """
    raise NotImplementedError


def count_splits_and_merges(
    gt: np.ndarray, pred: np.ndarray, overlap: float = 0.25
) -> tuple[int, int]:
    """Count over-segmentation (splits) and under-segmentation (merges).

    A split is one GT nucleus overlapping >=2 predictions by more than
    `overlap`; a merge is one prediction covering >=2 GT nuclei. Reported
    alongside AP because they distinguish opposite failure modes that AP alone
    conflates.
    """
    raise NotImplementedError
