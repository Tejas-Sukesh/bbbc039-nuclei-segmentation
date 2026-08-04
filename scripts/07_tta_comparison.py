#!/usr/bin/env python
"""Test-time augmentation: the deliberate optimization, with a stated prediction.

`augment=True` averages the network's output over flipped and rotated tiles.
It is the one remaining knob that changes the *representation* rather than the
post-processing, the cheap post-processing space having been shown optimal at its
defaults (see `grids.py`).

**The prediction, registered before running this.** The reasoning matters more
than the number, because either outcome is informative:

* Mean AP should rise, and the gain should be concentrated at IoU 0.85-0.95.
  Averaging over flips reduces *variance* in the predicted flow field, which
  moves boundaries by fractions of a pixel -- and over half the total AP
  shortfall sits at those two strictest thresholds.
* The merge count should stay near its baseline of 103. A merge is a *bias*: the
  network's learned representation puts one basin where the annotation puts two
  objects. Averaging eight flips of the same biased model reproduces the same
  merge more confidently rather than resolving it.

So if AP rises while merges hold, that is not a disappointing half-result -- it
is independent confirmation of the diagnosis in `failures.py`, that the errors
live upstream of anything post-processing can reach. If merges *do* fall
substantially, the diagnosis is wrong and this file says so.

Requires the augmented cache:

    python scripts/01_cache_flows.py --augment --splits validation test
    python scripts/07_tta_comparison.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from nucleiseg.data import split_names
from nucleiseg.evaluate import RESULTS, compare, evaluate_named, print_summary, save_results
from nucleiseg.segmenters import CellposeParams, FlowCache

PREDICTION = {
    "ap_rises": True,
    "gain_concentrated_at_strict_thresholds": True,
    "merges_hold_near_baseline": True,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = split_names(args.split)
    if args.limit:
        names = names[: args.limit]
    params = CellposeParams()

    caches = {
        "default": FlowCache(augment=False),
        "tta": FlowCache(augment=True),
    }
    for label, cache in caches.items():
        missing = [n for n in names if not cache.has(n)]
        if missing:
            raise SystemExit(
                f"{label}: {len(missing)} images not cached. Run "
                f"scripts/01_cache_flows.py{'  --augment' if label == 'tta' else ''} first."
            )

    aggs, scores = {}, {}
    for label, cache in caches.items():
        print(f"\nevaluating {label} on {args.split} ({len(names)} images)...")
        scores[label] = evaluate_named(lambda n, c=cache: c.masks(n, params), names)
        aggs[label] = save_results(
            f"cellpose_{label}_{args.split}",
            scores[label],
            params=params,
            extra={"augment": label == "tta"},
        )
        print_summary(f"cellpose_{label}_{args.split}", aggs[label])

    compare(f"default ({args.split})", aggs["default"], f"TTA ({args.split})", aggs["tta"])

    # Per-threshold, because "where the gain landed" is the whole prediction.
    d, t = aggs["default"]["per_threshold_macro"], aggs["tta"]["per_threshold_macro"]
    print("\n  gain by IoU threshold:")
    for k in sorted(d, key=float):
        print(f"    {float(k):.2f}   {d[k]:.4f} -> {t[k]:.4f}   ({t[k]-d[k]:+.4f})")

    strict = [k for k in d if float(k) >= 0.85]
    loose = [k for k in d if float(k) < 0.85]
    gain_strict = float(np.mean([t[k] - d[k] for k in strict]))
    gain_loose = float(np.mean([t[k] - d[k] for k in loose]))
    d_ap = aggs["tta"]["ap_macro"] - aggs["default"]["ap_macro"]
    d_merges = aggs["tta"]["merges_total"] - aggs["default"]["merges_total"]

    verdict = {
        "ap_rises": bool(d_ap > 0),
        "gain_concentrated_at_strict_thresholds": bool(gain_strict > gain_loose),
        "merges_hold_near_baseline": bool(
            abs(d_merges) <= 0.2 * aggs["default"]["merges_total"]
        ),
    }
    print("\n=== the registered prediction ===")
    for key, expected in PREDICTION.items():
        got = verdict[key]
        print(f"  {'HELD ' if got == expected else 'BROKE'}  {key}")
    print(f"\n  mean gain at IoU>=0.85 : {gain_strict:+.4f}")
    print(f"  mean gain at IoU<0.85  : {gain_loose:+.4f}")
    print(f"  merges  {aggs['default']['merges_total']} ->"
          f" {aggs['tta']['merges_total']}  ({d_merges:+d})")
    print(f"  splits  {aggs['default']['splits_total']} ->"
          f" {aggs['tta']['splits_total']}")
    if all(verdict[k] == v for k, v in PREDICTION.items()):
        print("\n  All three held: the gain is sub-pixel boundary placement, and the"
              "\n  merges are a bias that averaging cannot reach -- which is the"
              "\n  diagnosis in failures.py, confirmed by an intervention that had a"
              "\n  fair chance to refute it.")

    out = RESULTS / f"tta_comparison_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split,
        "n_images": len(names),
        "params": params.__dict__,
        "prediction": PREDICTION,
        "verdict": verdict,
        "ap_default": aggs["default"]["ap_macro"],
        "ap_tta": aggs["tta"]["ap_macro"],
        "ap_delta": d_ap,
        "ci_default": aggs["default"]["ap_macro_ci95"],
        "ci_tta": aggs["tta"]["ap_macro_ci95"],
        "gain_strict_thresholds": gain_strict,
        "gain_loose_thresholds": gain_loose,
        "per_threshold_default": d,
        "per_threshold_tta": t,
        "merges_default": aggs["default"]["merges_total"],
        "merges_tta": aggs["tta"]["merges_total"],
        "splits_default": aggs["default"]["splits_total"],
        "splits_tta": aggs["tta"]["splits_total"],
        "count_bias_default": aggs["default"]["count_bias_pct"],
        "count_bias_tta": aggs["tta"]["count_bias_pct"],
    }, indent=2, default=float))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
