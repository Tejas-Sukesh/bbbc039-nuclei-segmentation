"""Inventory of what the segmenter got wrong, object by object.

`metrics.py` answers *how much* is wrong. This answers *which objects*, and
attaches to each one the properties that let a failure be attributed to a
mechanism rather than to bad luck: how big it is, how bright, whether it touches
the field of view, and whether anything was predicted there at all.

The distinction that mechanism-hunting turns on is between a nucleus that was
**absorbed** into a neighbour and one that was **never detected**. Both are
false negatives and both cost the same in AP, but they need opposite fixes -- the
first is a separation problem in the flow field, the second a sensitivity problem
-- so an aggregate that lumps them together cannot point at either. The
distinguishing evidence is whether the missed nucleus overlaps a prediction at
all:

* `absorbed`   -- overlaps a prediction that also claims another nucleus. The
                  merge case; the object is inside a fused blob.
* `boundary`   -- overlaps a prediction, but too weakly to match and not shared.
                  The outline drifted rather than the object being lost.
* `undetected` -- no prediction overlaps it anywhere. Nothing was found here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from . import metrics as M


@dataclass(frozen=True)
class MissedObject:
    """One ground-truth nucleus with no match, and why it plausibly has none."""

    name: str
    gt_id: int
    area: int
    mean_intensity: float
    border_distance: int  # 0 == touching the edge of the field
    best_iou: float
    kind: str  # absorbed | boundary | undetected

    @property
    def touches_border(self) -> bool:
        return self.border_distance == 0


@dataclass(frozen=True)
class MergedPrediction:
    """One predicted object that covers several ground-truth nuclei.

    Carries the absorbed objects' areas because the *composition* of a merge is
    what identifies its mechanism, and the count alone hides it. Two nuclei of
    similar size fused together is a flow-field separation failure. One normal
    nucleus plus a much smaller object is something else entirely -- an
    annotation that calls a small bright punctum its own nucleus where the
    network calls it part of the parent.
    """

    name: str
    pred_id: int
    gt_ids: tuple[int, ...]
    gt_areas: tuple[int, ...]  # descending, so [0] is the dominant object
    area: int
    bbox: tuple[int, int, int, int]  # y0, y1, x0, x1

    @property
    def n_absorbed(self) -> int:
        return len(self.gt_ids)

    @property
    def kind(self) -> str:
        """`satellite`, `comparable`, or `mixed`, by relative size of what it fused."""
        if len(self.gt_areas) < 2:
            return "comparable"
        big, rest = self.gt_areas[0], self.gt_areas[1:]
        if all(r < 0.25 * big for r in rest):
            return "satellite"
        if all(r > 0.5 * big for r in rest):
            return "comparable"
        return "mixed"


def missed_objects(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    name: str = "",
    iou_threshold: float = 0.5,
    overlap: float = 0.25,
) -> list[MissedObject]:
    """Every unmatched ground-truth nucleus, classified by mechanism."""
    gt, n_gt = M.relabel_sequential(gt)
    pred, n_pred = M.relabel_sequential(pred)
    if n_gt == 0:
        return []
    if n_pred == 0:
        ious = np.zeros((n_gt, 0))
        matched: set[int] = set()
    else:
        ious = M.iou_matrix(gt, pred)
        matched = {g for g, _ in M.match_instances(ious, iou_threshold).matched}

    # Which predictions straddle two or more nuclei -- the same >25%-of-GT-area
    # rule `metrics.count_splits_and_merges` uses, so the two agree by
    # construction rather than by coincidence.
    if n_pred:
        hist = M._contingency(gt, pred)[0]
        inter = hist[1:, 1:].astype(np.float64)
        # Row sums must include the background column, or "fraction of the GT
        # object covered" is normalised by matched area instead of true area.
        gt_area = np.maximum(hist[1:, :].sum(axis=1, keepdims=True), 1)
        covering = (inter / gt_area) > overlap
        shared_pred = set(np.flatnonzero(covering.sum(axis=0) >= 2) + 1)
    else:
        inter, shared_pred = np.zeros((n_gt, 0)), set()

    h, w = gt.shape
    out = []
    for sl, gid in zip(ndi.find_objects(gt), range(1, n_gt + 1)):
        if gid in matched or sl is None:
            continue
        mask = gt[sl] == gid
        area = int(mask.sum())
        if area == 0:
            continue
        best_iou = float(ious[gid - 1].max()) if ious.shape[1] else 0.0

        if best_iou == 0.0:
            kind = "undetected"
        else:
            # Does the prediction covering it also claim someone else?
            claimants = set(np.flatnonzero(inter[gid - 1] > 0) + 1)
            kind = "absorbed" if claimants & shared_pred else "boundary"

        out.append(
            MissedObject(
                name=name,
                gt_id=gid,
                area=area,
                mean_intensity=float(image[sl][mask].mean()),
                border_distance=int(
                    min(sl[0].start, sl[1].start, h - sl[0].stop, w - sl[1].stop)
                ),
                best_iou=best_iou,
                kind=kind,
            )
        )
    return out


def merged_predictions(
    gt: np.ndarray, pred: np.ndarray, name: str = "", overlap: float = 0.25
) -> list[MergedPrediction]:
    """Predicted objects that fused several nuclei, worst (most absorbed) first."""
    gt, n_gt = M.relabel_sequential(gt)
    pred, n_pred = M.relabel_sequential(pred)
    if n_gt == 0 or n_pred == 0:
        return []
    hist = M._contingency(gt, pred)[0]
    inter = hist[1:, 1:].astype(np.float64)
    gt_area = np.maximum(hist[1:, :].sum(axis=1, keepdims=True), 1)
    covering = (inter / gt_area) > overlap

    out = []
    for sl, pid in zip(ndi.find_objects(pred), range(1, n_pred + 1)):
        gt_ids = np.flatnonzero(covering[:, pid - 1]) + 1
        if len(gt_ids) < 2 or sl is None:
            continue
        pairs = sorted(
            ((int(g), int(gt_area[g - 1, 0])) for g in gt_ids),
            key=lambda p: -p[1],
        )
        out.append(
            MergedPrediction(
                name=name,
                pred_id=pid,
                gt_ids=tuple(g for g, _ in pairs),
                gt_areas=tuple(a for _, a in pairs),
                area=int((pred[sl] == pid).sum()),
                bbox=(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop),
            )
        )
    return sorted(out, key=lambda m: -m.n_absorbed)


def summarize_merges(merges: list[MergedPrediction]) -> dict:
    """Break the merge count down by what each one actually fused.

    The distinction this exposes reorders the whole failure analysis: a repair
    layer that re-splits fused objects with a distance-transform watershed only
    addresses the `comparable` class, because a small punctum inside a nucleus is
    not a separate basin in the distance transform at all.
    """
    if not merges:
        return {"n_merges": 0}
    kinds = [m.kind for m in merges]
    secondary = np.array([a for m in merges for a in m.gt_areas[1:]])
    dominant = np.array([m.gt_areas[0] for m in merges])
    return {
        "n_merges": len(merges),
        "n_nuclei_absorbed": int(sum(m.n_absorbed for m in merges)),
        "worst_absorbed": int(max(m.n_absorbed for m in merges)),
        "by_kind": {k: kinds.count(k) for k in ("satellite", "comparable", "mixed")},
        "dominant_area_median": float(np.median(dominant)),
        "secondary_area_median": float(np.median(secondary)),
        "secondary_frac_under_100px": float(np.mean(secondary < 100)),
    }


def summarize_missed(missed: list[MissedObject], n_gt_total: int) -> dict:
    """Aggregate the inventory into the numbers the failure writeup quotes."""
    if not missed:
        return {"n_missed": 0, "n_gt_total": n_gt_total}
    area = np.array([m.area for m in missed])
    kinds = [m.kind for m in missed]
    return {
        "n_gt_total": n_gt_total,
        "n_missed": len(missed),
        "recall_at_50": (n_gt_total - len(missed)) / n_gt_total if n_gt_total else 0.0,
        "by_kind": {k: kinds.count(k) for k in ("undetected", "absorbed", "boundary")},
        "area_median": float(np.median(area)),
        "frac_under_100px": float(np.mean(area < 100)),
        "frac_under_min_size_15px": float(np.mean(area < 15)),
        "frac_touching_border": float(np.mean([m.touches_border for m in missed])),
        "intensity_median": float(np.median([m.mean_intensity for m in missed])),
    }
