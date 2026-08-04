#!/usr/bin/env python
"""Attack the measured failure mode: rescale the input so small objects resolve.

**Why this intervention and not another one.** Everything tuned so far --
`cellprob_threshold`, `min_size`, `flow_threshold`, `niter`, TTA -- operates
*downstream* of the network. The failure analysis concluded the error lives
upstream, in what the network resolves, which is exactly why all of them failed:
they were adjusting the output after the information was gone.

`diameter` is the first knob at the right level. Cellpose rescales the input by
`30 / diameter` before the network sees it (`models.py`: `rescale = 30./diameter`),
so it changes the *scale of the evidence*, not the interpretation of it.

**The quantitative case, from the model's own documentation.** The Cellpose
generalist models were trained on objects with diameters from **7.5 to 120 px,
mean 30 px**. In this dataset the median nucleus is 28.1 px across -- essentially
dead-on the training mean, which is why the defaults are hard to beat on typical
nuclei. But the 1st percentile is 3.9 px, and the median *missed* object is 20 px
in area, i.e. roughly 5 px across.

**Those objects are below the model's training range entirely.** Not badly
resolved -- out of distribution. Upscaling 2x (diameter=15) maps the dataset's
3.9-41.4 px spread onto 7.8-82.8 px, which lands the whole distribution inside
the trained range for the first time.

**The registered prediction, written before running.** Stated so the result can
refute it rather than be narrated afterwards:

1. **Recall on sub-100 px objects rises substantially.** This is the whole point;
   if it does not, the out-of-distribution explanation is wrong.
2. **Precision falls** -- upscaling lets the network resolve noise and debris as
   objects too, and nothing about this intervention distinguishes a real 5 px
   nucleus from a 5 px speck of debris.
3. **The merge count barely moves.** Merges are a bias in the learned
   representation; rescaling gives the network a bigger picture of the same
   evidence, and the 24 genuine touching-nuclei fusions have no reason to
   separate. (The 62 satellite "merges" *may* resolve, since those are small
   objects being absorbed -- so watch the two classes separately.)
4. **Net AP is genuinely uncertain.** Median nuclei go from 28 px (optimal) to
   56 px (still in range, but off-centre), so boundary precision on the typical
   nucleus may degrade while recall on the tail improves. Either direction is a
   result; the mechanism is what is being measured.

    python scripts/09_diameter.py --diameters 15 --split validation
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from nucleiseg import failures as F
from nucleiseg.data import load_sample, split_names
from nucleiseg.evaluate import RESULTS, evaluate_named, print_summary, save_results
from nucleiseg.segmenters import CellposeParams, FlowCache

PREDICTION = {
    "small_object_recall_rises": True,
    "precision_falls": True,
    "merges_hold_near_baseline": True,
}
TRAINED_DIAMETER_RANGE = (7.5, 120.0)


def small_object_stats(names, cache, params) -> dict:
    """Recall split by object size, plus the merge breakdown by kind."""
    missed, merges, n_gt = [], [], 0
    for name in names:
        s = load_sample(name)
        pred = cache.masks(name, params)
        n_gt += int(s.labels.max())
        missed += F.missed_objects(s.image, s.labels, pred, name=name)
        merges += F.merged_predictions(s.labels, pred, name=name)
    area = np.array([m.area for m in missed]) if missed else np.zeros(0)
    return {
        "n_gt": n_gt,
        "n_missed": len(missed),
        "recall_all": (n_gt - len(missed)) / n_gt,
        "n_missed_under_100px": int((area < 100).sum()),
        "n_missed_under_50px": int((area < 50).sum()),
        "merges": F.summarize_merges(merges),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--diameters", type=float, nargs="+", default=[15.0],
                    help="30/diameter is the upscale factor; 15 -> 2x")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = split_names(args.split)
    if args.limit:
        names = names[: args.limit]
    params = CellposeParams()

    runs = [("default", None)] + [(f"d{d:g}", d) for d in args.diameters]
    aggs, smalls = {}, {}

    for label, diam in runs:
        cache = FlowCache(diameter=diam)
        todo = [n for n in names if not cache.has(n)]
        if todo:
            scale = 30.0 / diam if diam else 1.0
            print(f"\n[{label}] caching {len(todo)} images at {scale:.2f}x "
                  f"(diameter={diam})...", flush=True)
            for i, n in enumerate(todo, 1):
                cache.build(n, load_sample(n).image)
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)

        print(f"\n[{label}] evaluating...")
        scores = evaluate_named(lambda n, c=cache: c.masks(n, params), names)
        aggs[label] = save_results(f"cellpose_{label}_{args.split}", scores,
                                   params=params, extra={"diameter": diam})
        print_summary(f"cellpose_{label}_{args.split}", aggs[label])
        smalls[label] = small_object_stats(names, cache, params)

    base, base_small = aggs["default"], smalls["default"]
    print("\n" + "=" * 66)
    print("does rescaling the input recover the small objects?")
    print("=" * 66)
    for label, diam in runs[1:]:
        a, s = aggs[label], smalls[label]
        scale = 30.0 / diam
        print(f"\n--- {label}  ({scale:.2f}x upscale, diameter={diam:g}) ---")
        print(f"  AP@[.5:.95]   {base['ap_macro']:.4f} -> {a['ap_macro']:.4f}"
              f"   ({a['ap_macro']-base['ap_macro']:+.4f})")
        lo_a, hi_a = base["ap_macro_ci95"]
        lo_b, hi_b = a["ap_macro_ci95"]
        if not (hi_a < lo_b or hi_b < lo_a):
            print("     (95% intervals overlap -- inside the noise for n=50)")
        print(f"  F1@0.5        {base['f1_50_macro']:.4f} -> {a['f1_50_macro']:.4f}")
        print(f"  recall        {base_small['recall_all']:.4f} -> {s['recall_all']:.4f}")
        print(f"  missed <100px {base_small['n_missed_under_100px']:4d} ->"
              f" {s['n_missed_under_100px']:4d}")
        print(f"  missed <50px  {base_small['n_missed_under_50px']:4d} ->"
              f" {s['n_missed_under_50px']:4d}")
        print(f"  objects found {base['n_pred_total']:5d} -> {a['n_pred_total']:5d}"
              f"   (GT {a['n_gt_total']})")
        print(f"  count bias    {base['count_bias_pct']:+.2f}% ->"
              f" {a['count_bias_pct']:+.2f}%")
        print(f"  merges        {base['merges_total']:4d} -> {a['merges_total']:4d}"
              f"     splits {base['splits_total']} -> {a['splits_total']}")
        bk, ak = base_small["merges"]["by_kind"], s["merges"]["by_kind"]
        print(f"    satellite   {bk['satellite']:4d} -> {ak['satellite']:4d}"
              f"     comparable {bk['comparable']} -> {ak['comparable']}")

        # False positives at IoU 0.5, as the precision side of the trade.
        fp_base = base["n_pred_total"] - (base_small["n_gt"] - base_small["n_missed"])
        fp_new = a["n_pred_total"] - (s["n_gt"] - s["n_missed"])
        print(f"  unmatched predictions (false positives)"
              f"  {fp_base:4d} -> {fp_new:4d}")

        verdict = {
            "small_object_recall_rises":
                s["n_missed_under_100px"] < 0.9 * base_small["n_missed_under_100px"],
            "precision_falls": fp_new > fp_base,
            "merges_hold_near_baseline":
                abs(a["merges_total"] - base["merges_total"]) <= 0.2 * base["merges_total"],
        }
        print("\n  registered prediction:")
        for k, expected in PREDICTION.items():
            print(f"    {'HELD ' if verdict[k] == expected else 'BROKE'}  {k}")

        out = RESULTS / f"diameter_{label}_{args.split}.json"
        out.write_text(json.dumps({
            "split": args.split, "diameter": diam, "upscale": scale,
            "trained_diameter_range": TRAINED_DIAMETER_RANGE,
            "prediction": PREDICTION, "verdict": verdict,
            "default": {"ap": base["ap_macro"], "f1_50": base["f1_50_macro"],
                        **{k: v for k, v in base_small.items() if k != "merges"},
                        "merges": base_small["merges"], "false_positives": fp_base},
            "rescaled": {"ap": a["ap_macro"], "f1_50": a["f1_50_macro"],
                         **{k: v for k, v in s.items() if k != "merges"},
                         "merges": s["merges"], "false_positives": fp_new},
        }, indent=2, default=float))
        print(f"\n  wrote {out.name}")


if __name__ == "__main__":
    main()
