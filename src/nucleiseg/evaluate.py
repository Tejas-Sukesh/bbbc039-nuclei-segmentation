"""Evaluation harness: run a segmenter over a split and persist the results.

Everything written here is designed so a number in the writeup can be traced
back to the exact code and parameters that produced it:

* **Per-image rows, always.** The failure analysis is driven by sorting images by
  score and looking at the worst ones, which a single aggregate makes impossible.
* **Parameters saved alongside the numbers.** A before/after comparison that
  cannot be re-run is not a result.
* **Both aggregations reported**, plus a bootstrap interval, because with ~50
  images a one-point difference is inside the noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import metrics as M
from .data import REPO_ROOT, load_sample, split_names

RESULTS = REPO_ROOT / "results"


def evaluate_named(
    predict: Callable[[str], np.ndarray],
    names: Iterable[str],
    progress: bool = False,
) -> list[M.ImageScore]:
    """Score a predictor over image stems. `predict(name) -> int32 labels`."""
    names = list(names)
    scores = []
    for i, name in enumerate(names, 1):
        sample = load_sample(name)
        scores.append(M.score_image(sample.labels, predict(name), name=name))
        if progress:
            print(f"  {i}/{len(names)} {name[:26]} ap={scores[-1].ap:.4f}", flush=True)
    return scores


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(
    tag: str,
    scores: list[M.ImageScore],
    params: object | None = None,
    extra: dict | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Write per-image CSV + summary JSON. Returns the aggregate dict."""
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = M.aggregate(scores)

    csv_path = out_dir / f"{tag}_per_image.csv"
    cols = ["name", "n_gt", "n_pred", "count_error", "ap", "f1_50",
            "mean_iou_matched", "splits", "merges"]
    with csv_path.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for s in sorted(scores, key=lambda s: s.ap):
            fh.write(
                f"{s.name},{s.n_gt},{s.n_pred},{s.count_error},{s.ap:.6f},"
                f"{s.f1_50:.6f},{s.mean_iou_matched:.6f},{s.splits},{s.merges}\n"
            )

    payload = {"tag": tag, "summary": _jsonable(agg), "params": _jsonable(params)}
    if extra:
        payload.update(_jsonable(extra))
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(payload, indent=2))
    return agg


def print_summary(tag: str, agg: dict) -> None:
    lo, hi = agg["ap_macro_ci95"]
    print(f"\n=== {tag} ({agg['n_images']} images) ===")
    print(f"  AP@[.5:.95] macro : {agg['ap_macro']:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  AP  micro          : {agg['ap_micro']:.4f}")
    print(f"  AP  macro, non-empty GT only : {agg['ap_macro_nonempty']:.4f}"
          f"   (empty GT fields: {agg['n_empty_gt']})")
    print(f"  F1@0.5 macro       : {agg['f1_50_macro']:.4f}")
    print(f"  mean IoU (matched) : {agg['mean_iou_matched']:.4f}")
    print(f"  splits / merges    : {agg['splits_total']} / {agg['merges_total']}")
    print(f"  nuclei GT vs pred  : {agg['n_gt_total']} vs {agg['n_pred_total']}"
          f"   count bias {agg['count_bias_pct']:+.2f}%")


def compare(tag_a: str, agg_a: dict, tag_b: str, agg_b: dict) -> None:
    """Print a before/after with the caveat the interval demands."""
    d = agg_b["ap_macro"] - agg_a["ap_macro"]
    lo_a, hi_a = agg_a["ap_macro_ci95"]
    lo_b, hi_b = agg_b["ap_macro_ci95"]
    print(f"\n=== {tag_a} -> {tag_b} ===")
    print(f"  AP {agg_a['ap_macro']:.4f} -> {agg_b['ap_macro']:.4f}   ({d:+.4f})")
    print(f"  count bias {agg_a['count_bias_pct']:+.2f}% -> {agg_b['count_bias_pct']:+.2f}%")
    print(f"  merges {agg_a['merges_total']} -> {agg_b['merges_total']},"
          f"  splits {agg_a['splits_total']} -> {agg_b['splits_total']}")
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    if overlap:
        print("  NOTE: the two 95% intervals overlap -- this gain is inside the noise "
              "for this sample size. Report it as such.")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate a segmenter on a BBBC039 split.")
    ap.add_argument("--split", default="validation", choices=["training", "validation", "test"])
    ap.add_argument("--segmenter", default="cellpose", choices=["cellpose", "classical"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N images only")
    args = ap.parse_args()

    names = split_names(args.split)
    if args.limit:
        names = names[: args.limit]
    tag = args.tag or f"{args.segmenter}_{args.split}"

    if args.segmenter == "cellpose":
        from .segmenters import CellposeParams, CellposeSegmenter

        seg = CellposeSegmenter(CellposeParams())
        predict, params = seg.segment_named, seg.params
    else:
        from .baseline import BaselineParams
        from .segmenters import ClassicalSegmenter

        seg = ClassicalSegmenter(BaselineParams())
        predict = lambda n: seg.segment(load_sample(n).image)  # noqa: E731
        params = seg.params

    print(f"Evaluating {args.segmenter} on {args.split} ({len(names)} images)...")
    scores = evaluate_named(predict, names, progress=True)
    agg = save_results(tag, scores, params=params)
    print_summary(tag, agg)
    print(f"\nWrote results/{tag}_per_image.csv and results/{tag}_summary.json")


if __name__ == "__main__":
    main()
