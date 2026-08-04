#!/usr/bin/env python
"""Two input-level interventions, chosen by measurement, tested against predictions.

Both knobs here change what the network *sees*, which is the level the failure
analysis pointed at and the level `09_diameter.py` showed actually moves things.
Every earlier attempt (cellprob_threshold, min_size, flow_threshold, niter, TTA)
adjusted the output instead, and all of them failed.

**What 09 established, and where it left off.** Upscaling 2x broke both of its
predictions, informatively:

* small-object misses did **not** move (265 -> 266 under 50 px), which kills the
  "too small for the network to resolve" explanation outright;
* merges dropped by a third (103 -> 69) and genuine two-nuclei fusions halved
  (24 -> 12), which was supposed to be a bias that rescaling could not reach.

So resolution is the lever for *merges*, not for missed small objects. The
question this script asks is what the small-object lever actually is.

**The measurement that motivates the second knob.** Comparing missed against
found objects *at matched size*, the missed ones are consistently 15-25% dimmer
relative to local background, in every size band including the largest:

    size band     missed contrast   found contrast   ratio
    15-50 px           204               254          0.80
    50-150 px          299               406          0.74
    150-400 px         389               460          0.84
    >400 px            388               460          0.84

Dimness is therefore an axis independent of size. And Cellpose normalises
brightness across the whole image by default, so a field with a few very bright
nuclei compresses everything dimmer toward background. `tile_norm_blocksize`
normalises within local tiles instead, judging each nucleus against its own
neighbourhood. Blocksize 128 is roughly 4x the 28 px median nucleus -- large
enough to contain several nuclei plus background, small enough to track
illumination variation across a field.

**Registered predictions, written before running.**

Local normalisation (`tn128`):
1. **Recall on dim objects rises**, and rises more than recall on bright ones --
   this is the mechanism claim, and the arm exists to test it.
2. **False positives rise**, because amplifying local contrast in genuinely empty
   regions amplifies noise into objects.
3. **Merges barely move.** Contrast is not what fuses two nuclei; 09 showed scale
   is.

Combined (`d15_tn128`):
4. **Best recall of the three arms**, because the two interventions address
   different failures -- merges via scale, faint objects via contrast.
5. **Worst false-positive count of the three**, for the same reason.
6. Net AP still uncertain: recall gains and false positives pull opposite ways,
   and 09's AP moved +0.002 while its error profile changed substantially.

    python scripts/10_input_level.py --split validation
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

ARMS = [
    ("default", {}),
    ("tn128", {"norm_blocksize": 128}),
    ("d15", {"diameter": 15.0}),
    ("d15_tn128", {"diameter": 15.0, "norm_blocksize": 128}),
]
PREDICTION = {
    "tn128": {"dim_recall_rises_more_than_bright": True, "false_positives_rise": True,
              "merges_hold": True},
    "d15_tn128": {"best_recall_of_all_arms": True, "most_false_positives": True},
}
DIM_QUANTILE = 0.33  # "dim" = bottom third by contrast above local background


def object_table(names):
    """Per-object size and contrast above local background, computed once."""
    out = {}
    for n in names:
        s = load_sample(n)
        gt, ng = M.relabel_sequential(s.labels)
        if ng == 0:
            out[n] = (gt, np.zeros(0))
            continue
        img = s.image.astype(np.float32)
        bg = float(np.median(img[gt == 0])) if (gt == 0).any() else 0.0
        contrast = np.array([float(img[gt == g].mean()) - bg for g in range(1, ng + 1)])
        out[n] = (gt, contrast)
    return out


def arm_stats(names, cache, params, table, dim_cut):
    """Recall overall and split by how dim the object is, plus errors."""
    missed, merges = [], []
    n_gt = n_pred = 0
    dim_tot = dim_missed = bright_tot = bright_missed = 0
    for n in names:
        s = load_sample(n)
        pred = cache.masks(n, params)
        gt, contrast = table[n]
        n_gt += int(gt.max())
        n_pred += int(pred.max())
        ms = F.missed_objects(s.image, s.labels, pred, name=n)
        missed += ms
        merges += F.merged_predictions(s.labels, pred, name=n)
        bad = {m.gt_id for m in ms}
        for g in range(1, int(gt.max()) + 1):
            is_dim = contrast[g - 1] < dim_cut
            if is_dim:
                dim_tot += 1
                dim_missed += g in bad
            else:
                bright_tot += 1
                bright_missed += g in bad
    return {
        "n_gt": n_gt, "n_pred": n_pred, "n_missed": len(missed),
        "recall": (n_gt - len(missed)) / n_gt,
        "recall_dim": (dim_tot - dim_missed) / max(dim_tot, 1),
        "recall_bright": (bright_tot - bright_missed) / max(bright_tot, 1),
        "false_positives": n_pred - (n_gt - len(missed)),
        "merges": F.summarize_merges(merges),
        "n_missed_under_50px": int(sum(1 for m in missed if m.area < 50)),
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

    print("measuring per-object contrast...")
    table = object_table(names)
    all_contrast = np.concatenate([c for _, c in table.values() if len(c)])
    dim_cut = float(np.quantile(all_contrast, DIM_QUANTILE))
    print(f"  'dim' = contrast below {dim_cut:.0f} (bottom {DIM_QUANTILE:.0%})")

    aggs, stats = {}, {}
    for label, kw in ARMS:
        cache = FlowCache(**kw)
        todo = [n for n in names if not cache.has(n)]
        if todo:
            print(f"\n[{label}] caching {len(todo)} images {kw}...", flush=True)
            for i, n in enumerate(todo, 1):
                cache.build(n, load_sample(n).image)
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
        print(f"\n[{label}] evaluating...")
        scores = evaluate_named(lambda n, c=cache: c.masks(n, params), names)
        aggs[label] = save_results(f"cellpose_{label}_{args.split}", scores,
                                   params=params, extra=kw)
        print_summary(f"cellpose_{label}_{args.split}", aggs[label])
        stats[label] = arm_stats(names, cache, params, table, dim_cut)

    print("\n" + "=" * 78)
    print(f"{'arm':<12}{'AP':>8}{'F1':>8}{'recall':>9}{'dim':>8}{'bright':>8}"
          f"{'FP':>7}{'merge':>7}{'split':>7}")
    print("=" * 78)
    for label, _ in ARMS:
        a, s = aggs[label], stats[label]
        print(f"{label:<12}{a['ap_macro']:>8.4f}{a['f1_50_macro']:>8.4f}"
              f"{s['recall']:>9.4f}{s['recall_dim']:>8.4f}{s['recall_bright']:>8.4f}"
              f"{s['false_positives']:>7d}{a['merges_total']:>7d}{a['splits_total']:>7d}")

    base, b = aggs["default"], stats["default"]
    verdict = {}
    t = stats["tn128"]
    verdict["tn128"] = {
        "dim_recall_rises_more_than_bright":
            (t["recall_dim"] - b["recall_dim"]) > (t["recall_bright"] - b["recall_bright"]),
        "false_positives_rise": t["false_positives"] > b["false_positives"],
        "merges_hold":
            abs(aggs["tn128"]["merges_total"] - base["merges_total"])
            <= 0.2 * base["merges_total"],
    }
    c = stats["d15_tn128"]
    verdict["d15_tn128"] = {
        "best_recall_of_all_arms": c["recall"] >= max(stats[l]["recall"] for l, _ in ARMS),
        "most_false_positives":
            c["false_positives"] >= max(stats[l]["false_positives"] for l, _ in ARMS),
    }
    print("\n=== registered predictions ===")
    for arm, preds in PREDICTION.items():
        print(f"  [{arm}]")
        for k, expected in preds.items():
            print(f"    {'HELD ' if verdict[arm][k] == expected else 'BROKE'}  {k}")

    print(f"\n  dim-object recall   {b['recall_dim']:.4f} -> tn128 {t['recall_dim']:.4f}"
          f" -> combined {c['recall_dim']:.4f}")
    print(f"  false positives     {b['false_positives']} -> {t['false_positives']}"
          f" -> {c['false_positives']}")

    out = RESULTS / f"input_level_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split, "dim_cut": dim_cut, "dim_quantile": DIM_QUANTILE,
        "prediction": PREDICTION, "verdict": verdict,
        "arms": {l: {"config": dict(kw), "ap": aggs[l]["ap_macro"],
                     "ap_ci95": aggs[l]["ap_macro_ci95"],
                     "f1_50": aggs[l]["f1_50_macro"],
                     "splits": aggs[l]["splits_total"],
                     "merges_total": aggs[l]["merges_total"],
                     "count_bias_pct": aggs[l]["count_bias_pct"], **stats[l]}
                 for l, kw in ARMS},
    }, indent=2, default=float))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
