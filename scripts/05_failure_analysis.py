#!/usr/bin/env python
"""Attribute the score to mechanisms: what is missed, and whose outline is right.

Two passes over a split, both reading the flow cache, neither touching a
parameter. They answer the two questions the aggregate score cannot:

1. **What gets missed, and by which mechanism** (`failures.py`) -- every
   unmatched nucleus, classified as absorbed into a neighbour, drifted at the
   boundary, or never detected at all. These need opposite fixes, so lumping
   them into one false-negative count points at nothing.

2. **Whose outline is closer to the truth** (`boundary.py`) -- over half of the
   total AP shortfall is spent at IoU 0.90 and 0.95, where "correct" means
   agreeing with a hand-drawn outline to within about half a pixel. This pass
   asks the image itself, via the half-maximum edge definition, which of the two
   outlines is better placed. If the prediction wins, the strict end of the
   metric is measuring annotation noise and no amount of tuning can recover it.

    python scripts/05_failure_analysis.py --split validation
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from nucleiseg import boundary as B
from nucleiseg import failures as F
from nucleiseg.data import load_sample, split_names
from nucleiseg.evaluate import RESULTS
from nucleiseg.segmenters import CellposeParams, FlowCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    names = split_names(args.split)
    if args.limit:
        names = names[: args.limit]
    tag = args.tag or f"failures_{args.split}"

    cache = FlowCache()
    missing = [n for n in names if not cache.has(n)]
    if missing:
        raise SystemExit(
            f"{len(missing)} images are not cached. Run scripts/01_cache_flows.py first."
        )

    params = CellposeParams()  # defaults: the honest "before" everywhere
    missed: list[F.MissedObject] = []
    merges: list[F.MergedPrediction] = []
    fits: list[B.BoundaryFit] = []
    n_gt_total = 0

    print(f"{len(names)} images, Cellpose defaults, reading the flow cache")
    for i, name in enumerate(names, 1):
        sample = load_sample(name)
        pred = cache.masks(name, params)
        n_gt_total += int(sample.labels.max())
        missed += F.missed_objects(sample.image, sample.labels, pred, name=name)
        merges += F.merged_predictions(sample.labels, pred, name=name)
        fits += B.fit_image(sample.image, sample.labels, pred, name=name)
        if i % 10 == 0 or i == len(names):
            print(
                f"  {i}/{len(names)}  missed={len(missed)}  merged={len(merges)}  "
                f"measurable outlines={len(fits)}",
                flush=True,
            )

    RESULTS.mkdir(parents=True, exist_ok=True)

    # ----- what gets missed --------------------------------------------------
    miss_summary = F.summarize_missed(missed, n_gt_total)
    csv_path = RESULTS / f"{tag}_missed_objects.csv"
    with csv_path.open("w") as fh:
        fh.write("name,gt_id,kind,area,mean_intensity,border_distance,best_iou\n")
        for m in sorted(missed, key=lambda m: m.area):
            fh.write(
                f"{m.name},{m.gt_id},{m.kind},{m.area},{m.mean_intensity:.1f},"
                f"{m.border_distance},{m.best_iou:.4f}\n"
            )

    merge_summary = F.summarize_merges(merges)
    merge_csv = RESULTS / f"{tag}_merged_predictions.csv"
    with merge_csv.open("w") as fh:
        fh.write("name,pred_id,kind,n_absorbed,area,gt_ids,gt_areas,y0,y1,x0,x1\n")
        for m in merges:
            y0, y1, x0, x1 = m.bbox
            fh.write(
                f"{m.name},{m.pred_id},{m.kind},{m.n_absorbed},{m.area},"
                f"\"{' '.join(str(g) for g in m.gt_ids)}\","
                f"\"{' '.join(str(a) for a in m.gt_areas)}\",{y0},{y1},{x0},{x1}\n"
            )

    print(f"\n=== what gets missed ({args.split}) ===")
    print(f"  ground-truth nuclei      : {n_gt_total}")
    print(f"  missed at IoU 0.5        : {miss_summary['n_missed']}"
          f"   (recall {miss_summary['recall_at_50']:.1%})")
    for kind, n in miss_summary["by_kind"].items():
        print(f"    {kind:11s}          : {n:4d}  ({n/miss_summary['n_missed']:.0%})")
    print(f"  median missed area       : {miss_summary['area_median']:.0f} px")
    print(f"  under 100 px             : {miss_summary['frac_under_100px']:.0%}")
    print(f"  under min_size (15 px)   : {miss_summary['frac_under_min_size_15px']:.0%}")
    print(f"  touching the field edge  : {miss_summary['frac_touching_border']:.0%}")
    print(f"\n=== what the merges actually fused ===")
    print(f"  predictions fusing >=2 nuclei : {merge_summary['n_merges']}"
          f"   (absorbing {merge_summary['n_nuclei_absorbed']} nuclei)")
    for kind, n in merge_summary["by_kind"].items():
        note = {
            "satellite": "one normal nucleus + a much smaller object",
            "comparable": "nuclei of similar size fused -- the flow-field failure",
            "mixed": "neither cleanly",
        }[kind]
        print(f"    {kind:11s} : {n:4d}  ({n/merge_summary['n_merges']:.0%})  {note}")
    print(f"  dominant object median  : {merge_summary['dominant_area_median']:.0f} px")
    print(f"  absorbed object median  : {merge_summary['secondary_area_median']:.0f} px"
          f"   ({merge_summary['secondary_frac_under_100px']:.0%} under 100 px)")

    # ----- whose outline is right -------------------------------------------
    b_summary = B.summarize(fits)
    fit_csv = RESULTS / f"{tag}_boundary_fits.csv"
    with fit_csv.open("w") as fh:
        fh.write("name,gt_id,pred_id,area,contrast,gt_level,pred_level,gt_grad,pred_grad\n")
        for f in fits:
            fh.write(
                f"{f.name},{f.gt_id},{f.pred_id},{f.area},{f.contrast:.1f},"
                f"{f.gt_level:.4f},{f.pred_level:.4f},{f.gt_grad:.6f},{f.pred_grad:.6f}\n"
            )

    print(f"\n=== whose outline sits where the image says the edge is ===")
    print(f"  measurable nuclei        : {b_summary['n_objects']}"
          f" across {b_summary['n_images']} images")
    print("  (0.5 = half-maximum, the sub-pixel edge; >0.5 traced tight, <0.5 traced loose)")
    print(f"  ground truth level       : {b_summary['gt_level_median']:.3f}"
          f"   offset {b_summary['gt_abs_error_median']:.3f}")
    print(f"  prediction level         : {b_summary['pred_level_median']:.3f}"
          f"   offset {b_summary['pred_abs_error_median']:.3f}")
    print(f"  prediction closer        : {b_summary['pred_closer_fraction']:.1%} of nuclei")
    print(f"  prediction sharper edge  : {b_summary['pred_sharper_fraction']:.1%} of nuclei"
          f"  (independent gradient check)")
    print(f"\n  {B.verdict(b_summary)}")

    payload = {
        "tag": tag,
        "split": args.split,
        "params": params.__dict__,
        "missed": miss_summary,
        "merges": merge_summary,
        "annotation_ceiling": b_summary,
        "annotation_ceiling_verdict": B.verdict(b_summary),
    }
    out = RESULTS / f"{tag}_summary.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out.name}, {csv_path.name}, {merge_csv.name}, {fit_csv.name}")


if __name__ == "__main__":
    main()
