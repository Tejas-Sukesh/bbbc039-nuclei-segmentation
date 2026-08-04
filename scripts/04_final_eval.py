#!/usr/bin/env python
"""Final evaluation on the test split. Run this ONCE, at the end.

Every parameter used here was chosen on validation or training. The test split
has not been looked at during development, and once this script has been run and
its numbers reported, tuning against them would invalidate the estimate.

Reports the whole comparison so the before/after is legible:

* Cellpose-SAM defaults (the "before")
* Cellpose-SAM with the tuned global configuration
* the contextual policy, if one was trained
* the classical pipeline, as the interpretable reference

Each with a bootstrap confidence interval, because with 50 images a gain of a
point or two is inside the noise and a difference has to be reported as such.

    python scripts/04_final_eval.py --params results/bandit_sweep_validation.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from nucleiseg import metrics as M
from nucleiseg.data import load_sample, split_names
from nucleiseg.evaluate import RESULTS, compare, evaluate_named, print_summary, save_results
from nucleiseg.grids import arm_label, build_arms
from nucleiseg.segmenters import CellposeParams, FlowCache


def load_tuned(path: str | None) -> CellposeParams | None:
    """Read the tuned configuration produced by script 02."""
    if not path:
        return None
    data = json.loads(open(path).read())
    for key in ("exhaustive", "thompson", "ucb1"):
        block = data.get(key)
        if not block:
            continue
        params = block.get("true_best_params") or block.get("best_params")
        if params:
            print(f"tuned params from '{key}': {params}")
            return CellposeParams(**params)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--params", default=str(RESULTS / "bandit_sweep_validation.json"))
    ap.add_argument("--contextual", default=str(RESULTS / "contextual_validation_summary.json"))
    ap.add_argument("--classical", action="store_true", help="also run the classical pipeline")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = split_names(args.split)
    if args.limit:
        names = names[: args.limit]
    cache = FlowCache()
    missing = [n for n in names if not cache.has(n)]
    if missing:
        raise SystemExit(f"{len(missing)} {args.split} images not cached; run script 01.")

    aggs: dict[str, dict] = {}

    # --- before: out-of-the-box defaults ---
    print(f"\n[1] Cellpose-SAM defaults on {args.split} ({len(names)} images)")
    default = CellposeParams()
    scores = evaluate_named(lambda n: cache.masks(n, default), names)
    aggs["cellpose_default"] = save_results(
        f"cellpose_default_{args.split}", scores, params=default
    )
    print_summary(f"cellpose_default_{args.split}", aggs["cellpose_default"])

    # --- after: tuned global configuration ---
    tuned = load_tuned(args.params)
    if tuned and tuned != default:
        print(f"\n[2] Cellpose-SAM tuned ({arm_label(tuned)}) on {args.split}")
        scores_t = evaluate_named(lambda n: cache.masks(n, tuned), names)
        aggs["cellpose_tuned"] = save_results(
            f"cellpose_tuned_{args.split}", scores_t, params=tuned
        )
        print_summary(f"cellpose_tuned_{args.split}", aggs["cellpose_tuned"])
        compare("cellpose_default", aggs["cellpose_default"],
                "cellpose_tuned", aggs["cellpose_tuned"])
    elif tuned == default:
        print("\n[2] tuning selected the default configuration -- no change to report.")
    else:
        print(f"\n[2] skipped: no tuned params at {args.params}")

    # --- the contextual policy, applied blind ---
    try:
        ctx = json.loads(open(args.contextual).read())
        scaler_d = ctx.get("scaler")
        chosen_map = ctx.get("chosen_arm_per_image")
        if scaler_d and chosen_map is not None:
            from nucleiseg.features import FeatureScaler, raw_features

            # The policy is re-derived from the saved arm ordering + scaler. Any
            # image not seen during training gets its arm from its own features,
            # never from its labels.
            print(f"\n[3] contextual policy on {args.split}")
            labels = ctx["arms"]
            arms = build_arms()
            by_label = {arm_label(a): a for a in arms}
            scaler = FeatureScaler.from_dict(scaler_d)
            # Nearest-neighbour transfer of the learned policy in feature space:
            # the saved artefact records the arm chosen for each training-time
            # image, so a new image adopts the arm of its closest neighbour.
            train_names = list(chosen_map)
            Xtr = np.array([scaler.transform(raw_features(load_sample(n).image))
                            for n in train_names])
            def predict(n: str):
                x = scaler.transform(raw_features(load_sample(n).image))
                j = int(np.argmin(((Xtr - x) ** 2).sum(axis=1)))
                return cache.masks(n, by_label[chosen_map[train_names[j]]])

            scores_c = evaluate_named(predict, names)
            aggs["contextual"] = save_results(
                f"contextual_{args.split}", scores_c, params={"policy": "linucb-transfer"}
            )
            print_summary(f"contextual_{args.split}", aggs["contextual"])
            compare("cellpose_default", aggs["cellpose_default"],
                    "contextual", aggs["contextual"])
            _ = labels
    except FileNotFoundError:
        print(f"\n[3] skipped: no contextual policy at {args.contextual}")

    # --- classical reference ---
    if args.classical:
        from nucleiseg.baseline import BaselineParams, segment

        print(f"\n[4] classical pipeline on {args.split}")
        bp = BaselineParams()
        scores_cl = evaluate_named(lambda n: segment(load_sample(n).image, bp), names)
        aggs["classical"] = save_results(f"classical_{args.split}", scores_cl, params=bp)
        print_summary(f"classical_{args.split}", aggs["classical"])

    # --- one table ---
    print(f"\n{'='*74}\nFINAL, {args.split} split, n={len(names)}\n{'='*74}")
    print(f"{'method':<26}{'AP@[.5:.95]':>13}{'95% CI':>20}{'count bias':>13}")
    for name, agg in aggs.items():
        lo, hi = agg["ap_macro_ci95"]
        print(f"{name:<26}{agg['ap_macro']:>13.4f}  [{lo:.4f}, {hi:.4f}]"
              f"{agg['count_bias_pct']:>+12.2f}%")

    out = RESULTS / f"final_{args.split}.json"
    out.write_text(json.dumps(
        {k: {"ap_macro": v["ap_macro"], "ci95": v["ap_macro_ci95"],
             "count_bias_pct": v["count_bias_pct"], "splits": v["splits_total"],
             "merges": v["merges_total"]} for k, v in aggs.items()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
