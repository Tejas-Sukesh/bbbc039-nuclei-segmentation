#!/usr/bin/env python
"""Second pass over the residual: does it recover the small faint objects?

The first method here that is *built* rather than tuned. Everything before this
adjusted a parameter of someone else's model; this adds a stage, and it exists
because the failure analysis said exactly what was missing and why every global
knob failed to get it.

**The registered predictions**, written before running:

1. **Recall on objects under 50 px rises substantially.** This is the entire
   point. The residual makes those objects locally prominent, so a scale-selective
   detector should find a large fraction of them.
2. **False positives rise**, because some background structure will pass the
   contrast and edge-ratio gates.
3. **Mean AP barely moves, and may fall.** The recovered objects are tiny, so each
   contributes a matched object at loose IoU but is unlikely to survive strict
   thresholds; meanwhile every false positive costs at all ten. AP is the wrong
   metric for this intervention, which is itself worth stating.
4. **F1@0.5 is the metric that should move**, since it weighs the recall gain
   against the precision cost at the one threshold where small objects can match.

If (1) fails, the objects are not recoverable from the image at all and the
remaining explanation is that they are below the sensor noise floor.

    python scripts/11_residual_pass.py --split validation
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from nucleiseg import failures as F
from nucleiseg import metrics as M
from nucleiseg.data import load_sample, split_names
from nucleiseg.evaluate import RESULTS, evaluate_named, print_summary, save_results
from nucleiseg.segmenters import CellposeParams, FlowCache
from nucleiseg.smallobj import ResidualParams, augment_labels

PREDICTION = {
    "small_recall_rises": True,
    "false_positives_rise": True,
    "f1_50_rises": True,
}


def size_recall(names, predict, cut: int = 50) -> dict:
    """Recall split at `cut` px, plus false positives, from one prediction fn."""
    small_tot = small_hit = big_tot = big_hit = 0
    n_gt = n_pred = n_missed = 0
    for name in names:
        s = load_sample(name)
        gt, ng = M.relabel_sequential(s.labels)
        pred = predict(name)
        n_gt += ng
        n_pred += int(pred.max())
        if ng == 0:
            continue
        missed = {m.gt_id for m in F.missed_objects(s.image, s.labels, pred, name=name)}
        n_missed += len(missed)
        areas = np.bincount(gt.ravel())[1:]
        for g in range(1, ng + 1):
            if areas[g - 1] < cut:
                small_tot += 1
                small_hit += g not in missed
            else:
                big_tot += 1
                big_hit += g not in missed
    return {
        "n_gt": n_gt, "n_pred": n_pred, "n_missed": n_missed,
        "recall": (n_gt - n_missed) / n_gt,
        "recall_small": small_hit / max(small_tot, 1),
        "recall_large": big_hit / max(big_tot, 1),
        "n_small": small_tot,
        "false_positives": n_pred - (n_gt - n_missed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None, help="LoG threshold")
    ap.add_argument("--min-rel-contrast", type=float, default=None)
    args = ap.parse_args()

    names = split_names(args.split)
    if args.limit:
        names = names[: args.limit]

    rp = ResidualParams()
    if args.threshold is not None:
        rp = ResidualParams(**{**rp.__dict__, "threshold": args.threshold})
    if args.min_rel_contrast is not None:
        rp = ResidualParams(**{**rp.__dict__, "min_rel_contrast": args.min_rel_contrast})

    cache = FlowCache()
    params = CellposeParams()
    missing = [n for n in names if not cache.has(n)]
    if missing:
        raise SystemExit(f"{len(missing)} images not cached; run scripts/01_cache_flows.py")

    base = lambda n: cache.masks(n, params)  # noqa: E731
    added_total = {"n": 0}

    def augmented(name):
        lab, added = augment_labels(load_sample(name).image, base(name), rp)
        added_total["n"] += added
        return lab

    print(f"{len(names)} images, {args.split}\nparams: {rp}\n")
    print("[before] first pass only")
    a_before = save_results(f"cellpose_default_{args.split}",
                            evaluate_named(base, names), params=params)
    print_summary("before", a_before)
    s_before = size_recall(names, base)

    print("\n[after] first pass + residual second pass")
    a_after = save_results(f"residual_{args.split}",
                           evaluate_named(augmented, names), params=params,
                           extra={"residual_params": rp.__dict__})
    print_summary("after", a_after)
    s_after = size_recall(names, augmented)

    print("\n" + "=" * 62)
    print(f"objects added by the second pass : {added_total['n'] // 2}")
    print(f"{'':32}{'before':>10}{'after':>10}")
    rows = [
        ("recall, objects < 50 px", s_before["recall_small"], s_after["recall_small"], "%"),
        ("recall, objects >= 50 px", s_before["recall_large"], s_after["recall_large"], "%"),
        ("recall, all", s_before["recall"], s_after["recall"], "%"),
        ("F1@0.5", a_before["f1_50_macro"], a_after["f1_50_macro"], "f"),
        ("AP@[.5:.95]", a_before["ap_macro"], a_after["ap_macro"], "f"),
    ]
    for label, b, a, kind in rows:
        fmt = (lambda v: f"{v:.1%}") if kind == "%" else (lambda v: f"{v:.4f}")
        print(f"  {label:30}{fmt(b):>10}{fmt(a):>10}")
    print(f"  {'false positives':30}{s_before['false_positives']:>10}"
          f"{s_after['false_positives']:>10}")
    print(f"  {'objects predicted':30}{a_before['n_pred_total']:>10}"
          f"{a_after['n_pred_total']:>10}   (GT {a_after['n_gt_total']})")

    verdict = {
        "small_recall_rises": s_after["recall_small"] > s_before["recall_small"] + 0.02,
        "false_positives_rise": s_after["false_positives"] > s_before["false_positives"],
        "f1_50_rises": a_after["f1_50_macro"] > a_before["f1_50_macro"],
    }
    print("\n=== registered predictions ===")
    for k, expected in PREDICTION.items():
        print(f"  {'HELD ' if verdict[k] == expected else 'BROKE'}  {k}")

    out = RESULTS / f"residual_pass_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split, "params": rp.__dict__,
        "prediction": PREDICTION, "verdict": verdict,
        "before": {**s_before, "ap": a_before["ap_macro"],
                   "f1_50": a_before["f1_50_macro"]},
        "after": {**s_after, "ap": a_after["ap_macro"],
                  "f1_50": a_after["f1_50_macro"],
                  "ap_ci95": a_after["ap_macro_ci95"]},
    }, indent=2, default=float))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
