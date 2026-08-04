"""Instance-segmentation metrics against ground-truth labels.

This module is the contract the rest of the project is measured by, so the
conventions are stated explicitly rather than inherited from a library.

Conventions
-----------
* Label images are integer arrays, 0 = background, positive = instance IDs.
  IDs need not be contiguous; they are relabelled internally.
* Predicted/GT instances are matched **one-to-one by IoU**, greedily in
  descending IoU order. For thresholds >= 0.5 greedy matching is provably
  optimal, because a prediction cannot exceed IoU 0.5 with two disjoint GT
  objects, so the assignment is unique. `strategy="hungarian"` is available to
  verify that claim empirically.
* The headline score is **mean average precision over IoU 0.50:0.05:0.95**, the
  Data Science Bowl 2018 convention. Note that this "AP" is
  `TP / (TP + FP + FN)` at each threshold -- a Jaccard-style ratio over
  *objects*. It is NOT the area under a precision-recall curve, despite sharing
  the name with detection AP. Detection AP is not even computable here: it
  requires a per-object confidence score, which a watershed pipeline does not
  produce.
* **Empty fields.** Three BBBC039 fields contain zero nuclei, so 0/0 is
  reachable. Convention: predicting nothing on an empty field scores 1.0 (a
  correct answer scores as correct); predicting anything scores 0.0. Non-empty
  GT with an empty prediction scores 0.0. This is a choice, not a fact -- see
  `aggregate` for why it is also reported both with and without those fields.
* **Splits and merges are counted separately** from AP. A split (one GT cut into
  two) and a merge (two GT fused into one) cost nearly the same in AP but are
  opposite errors needing opposite fixes, and they cancel exactly in the object
  count -- so a pipeline can report a perfect count with every object wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.05), 2)


def relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Map arbitrary positive IDs onto 1..N, preserving background as 0.

    Needed because `iou_matrix` indexes a histogram by label value, so gaps in
    the ID sequence would allocate empty rows and shift the indexing. Cellpose
    output in particular is not guaranteed contiguous after size filtering.
    """
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros_like(labels, dtype=np.int32), 0
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    lookup[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    return lookup[labels], int(ids.size)


def _contingency(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Joint histogram of (gt_id, pred_id) pixel pairs, shape (n_gt+1, n_pred+1)."""
    gt, n_gt = relabel_sequential(gt)
    pred, n_pred = relabel_sequential(pred)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt + 1, n_pred + 1), dtype=np.int64), n_gt, n_pred
    # One pass: encode each pixel's (gt, pred) pair as a single integer, then
    # bincount. Looping over instance pairs would be O(n_gt * n_pred) full-image
    # comparisons -- far too slow at ~120 nuclei per field over 50 fields.
    flat = gt.astype(np.int64).ravel() * (n_pred + 1) + pred.astype(np.int64).ravel()
    counts = np.bincount(flat, minlength=(n_gt + 1) * (n_pred + 1))
    return counts.reshape(n_gt + 1, n_pred + 1), n_gt, n_pred


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pairwise IoU between every GT and predicted instance, shape (n_gt, n_pred)."""
    hist, n_gt, n_pred = _contingency(gt, pred)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64)
    intersection = hist[1:, 1:].astype(np.float64)
    gt_area = hist[1:, :].sum(axis=1, keepdims=True)  # includes bg overlap
    pred_area = hist[:, 1:].sum(axis=0, keepdims=True)
    union = gt_area + pred_area - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


@dataclass
class MatchResult:
    """One-to-one assignment between GT and predicted instances at one threshold."""

    threshold: float
    n_gt: int
    n_pred: int
    matched: list[tuple[int, int]] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return len(self.matched)

    @property
    def fn(self) -> int:
        return self.n_gt - self.tp

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else 1.0

    @property
    def dsb_score(self) -> float:
        """TP / (TP + FP + FN) -- the Kaggle DSB2018 per-threshold score."""
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 1.0


def match_instances(
    ious: np.ndarray, threshold: float, strategy: str = "greedy"
) -> MatchResult:
    """Match GT to predictions one-to-one at an IoU threshold."""
    n_gt, n_pred = ious.shape
    result = MatchResult(threshold=float(threshold), n_gt=n_gt, n_pred=n_pred)
    if n_gt == 0 or n_pred == 0:
        return result

    if strategy == "greedy":
        candidates = np.argwhere(ious >= threshold)
        if candidates.size:
            order = np.argsort(-ious[candidates[:, 0], candidates[:, 1]])
            used_gt: set[int] = set()
            used_pred: set[int] = set()
            for g, p in candidates[order]:
                if g not in used_gt and p not in used_pred:
                    used_gt.add(int(g))
                    used_pred.add(int(p))
                    result.matched.append((int(g) + 1, int(p) + 1))
    elif strategy == "hungarian":
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-ious)
        result.matched = [
            (int(r) + 1, int(c) + 1)
            for r, c in zip(rows, cols)
            if ious[r, c] >= threshold
        ]
    else:
        raise ValueError(f"strategy must be 'greedy' or 'hungarian', got {strategy!r}")

    return result


def count_splits_and_merges(
    gt: np.ndarray, pred: np.ndarray, overlap: float = 0.25
) -> tuple[int, int]:
    """Count over-segmentation (splits) and under-segmentation (merges).

    A **split** is one GT nucleus that at least two predictions each cover by
    more than `overlap` of the GT area. A **merge** is one prediction that
    covers at least two GT nuclei, each by more than `overlap` of that GT's
    area.
    """
    hist, n_gt, n_pred = _contingency(gt, pred)
    if n_gt == 0 or n_pred == 0:
        return 0, 0
    inter = hist[1:, 1:].astype(np.float64)
    gt_area = hist[1:, :].sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_of_gt = np.where(gt_area > 0, inter / gt_area, 0.0)
    covering = frac_of_gt > overlap
    splits = int((covering.sum(axis=1) >= 2).sum())  # per GT: >=2 preds cover it
    merges = int((covering.sum(axis=0) >= 2).sum())  # per pred: covers >=2 GT
    return splits, merges


@dataclass
class ImageScore:
    """Everything measured for a single field of view."""

    name: str
    n_gt: int
    n_pred: int
    ap: float  # mean over IOU_THRESHOLDS
    per_threshold: dict[float, float]
    f1_50: float
    mean_iou_matched: float  # boundary quality of the objects we did find
    splits: int
    merges: int

    @property
    def count_error(self) -> int:
        return self.n_pred - self.n_gt


def score_image(
    gt: np.ndarray,
    pred: np.ndarray,
    name: str = "",
    strategy: str = "greedy",
    overlap: float = 0.25,
) -> ImageScore:
    """Score one predicted label image against ground truth.

    This is the single function the optimizers use as a reward. Keep it cheap
    and side-effect free: the bandits call it thousands of times.
    """
    _, n_gt = relabel_sequential(gt)
    _, n_pred = relabel_sequential(pred)

    # Degenerate cases, per the convention documented at module level.
    if n_gt == 0 or n_pred == 0:
        score = 1.0 if (n_gt == 0 and n_pred == 0) else 0.0
        return ImageScore(
            name=name,
            n_gt=n_gt,
            n_pred=n_pred,
            ap=score,
            per_threshold={float(t): score for t in IOU_THRESHOLDS},
            f1_50=score,
            mean_iou_matched=float("nan"),
            splits=0,
            merges=0,
        )

    ious = iou_matrix(gt, pred)
    per_threshold, f1_50, matched_ious = {}, 0.0, []
    for t in IOU_THRESHOLDS:
        m = match_instances(ious, float(t), strategy=strategy)
        per_threshold[float(t)] = m.dsb_score
        if np.isclose(t, 0.50):
            f1_50 = m.f1
            matched_ious = [ious[g - 1, p - 1] for g, p in m.matched]

    splits, merges = count_splits_and_merges(gt, pred, overlap=overlap)
    return ImageScore(
        name=name,
        n_gt=n_gt,
        n_pred=n_pred,
        ap=float(np.mean(list(per_threshold.values()))),
        per_threshold=per_threshold,
        f1_50=f1_50,
        mean_iou_matched=float(np.mean(matched_ious)) if matched_ious else float("nan"),
        splits=splits,
        merges=merges,
    )


def bootstrap_ci(
    values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-image scores.

    Needed because a split has only ~50 images: differences of a point or two
    in mean AP are inside the noise, and a before/after claim without an
    interval is not a claim.
    """
    values = np.asarray(list(values), dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def aggregate(scores: list[ImageScore], seed: int = 0) -> dict:
    """Summarise per-image scores into the numbers that go in the writeup.

    Reports both aggregations deliberately, because they answer different
    questions and disagree in a way that matters here:

    * **macro** -- mean over images. Keeps sparse and degenerate fields visible,
      but hands each of ~50 images 2% of the score.
    * **micro** -- pool TP/FP/FN over all images, then compute. Robust to
      degenerate fields with no convention needed, but lets dense fields
      dominate.

    Also reports macro excluding empty-GT fields, so the reader can see exactly
    how much the empty-field convention is worth.
    """
    if not scores:
        return {}
    aps = np.array([s.ap for s in scores])
    nonempty = [s for s in scores if s.n_gt > 0]
    lo, hi = bootstrap_ci(aps, seed=seed)

    # Micro: recompute pooled counts at each threshold from per-image TP/FP/FN.
    # dsb_score = TP/(TP+FP+FN), so recover TP per threshold from the identity.
    micro_per_t = {}
    for t in IOU_THRESHOLDS:
        tp = fp = fn = 0
        for s in scores:
            if s.n_gt == 0 and s.n_pred == 0:
                continue
            r = s.per_threshold[float(t)]
            # TP = r*(TP+FP+FN); with FP = n_pred-TP and FN = n_gt-TP this gives
            # TP = r*(n_gt + n_pred - TP)  =>  TP = r*(n_gt+n_pred)/(1+r)
            tp_i = r * (s.n_gt + s.n_pred) / (1 + r) if r > 0 else 0.0
            tp += tp_i
            fp += s.n_pred - tp_i
            fn += s.n_gt - tp_i
        micro_per_t[float(t)] = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0

    total_gt = sum(s.n_gt for s in scores)
    total_pred = sum(s.n_pred for s in scores)
    return {
        "n_images": len(scores),
        "ap_macro": float(aps.mean()),
        "ap_macro_ci95": (lo, hi),
        "ap_macro_nonempty": float(np.mean([s.ap for s in nonempty])) if nonempty else float("nan"),
        "ap_micro": float(np.mean(list(micro_per_t.values()))),
        "per_threshold_macro": {
            float(t): float(np.mean([s.per_threshold[float(t)] for s in scores]))
            for t in IOU_THRESHOLDS
        },
        "f1_50_macro": float(np.mean([s.f1_50 for s in scores])),
        "mean_iou_matched": float(np.nanmean([s.mean_iou_matched for s in scores])),
        "splits_total": sum(s.splits for s in scores),
        "merges_total": sum(s.merges for s in scores),
        "n_gt_total": total_gt,
        "n_pred_total": total_pred,
        "count_bias_pct": 100.0 * (total_pred - total_gt) / total_gt if total_gt else 0.0,
        "n_empty_gt": sum(1 for s in scores if s.n_gt == 0),
    }
