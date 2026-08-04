#!/usr/bin/env python
"""Learn a per-image parameter policy with a contextual bandit (LinUCB).

The hypothesis: one global parameter set is a compromise across fields that do
not want the same compromise. Measured support for that -- per-image AP ranges
from about 0.70 to 0.87, and the worst field in a 12-image sample was a sparse
one with 43 nuclei while dense fields behave completely differently.

So instead of "which configuration is best on average," ask "which configuration
is best for an image that *looks* like this," using only features computable
from the raw image.

Four numbers are reported, and the fourth is what makes this honest:

1. **default** -- Cellpose-SAM out of the box. The "before".
2. **global best** -- the single best configuration, tuned on training.
3. **contextual policy** -- LinUCB's learned greedy policy.
4. **per-image oracle** -- the best arm for each image, chosen *with* ground
   truth. This is unattainable, and it is the ceiling on what any per-image
   adaptation could ever achieve. If the oracle barely beats the global best,
   then there is no headroom for a contextual method and the correct conclusion
   is that the hypothesis was wrong. Reporting that would be a real result.

Training uses the training split. Evaluation uses validation. The test split is
untouched here.

**No label leakage**: the policy maps image features -> arm. Ground truth is used
to compute the reward during *training*, and to compute the oracle for reference,
but never to choose parameters for an image being scored by the policy.

    python scripts/03_contextual_bandit.py --rounds 4000
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from nucleiseg import metrics as M
from nucleiseg.bandits import LinUCB
from nucleiseg.data import load_sample, split_names
from nucleiseg.evaluate import RESULTS, print_summary, save_results
from nucleiseg.features import FEATURE_NAMES, FeatureScaler, raw_features
from nucleiseg.grids import arm_label, build_arms
from nucleiseg.segmenters import CellposeParams, FlowCache


def reward_matrix(cache: FlowCache, names, arms, gts) -> np.ndarray:
    """Full (n_images, n_arms) AP matrix. Cheap because flows are cached."""
    R = np.zeros((len(names), len(arms)), dtype=np.float64)
    for i, name in enumerate(names):
        for j, arm in enumerate(arms):
            R[i, j] = M.score_image(gts[name], cache.masks(name, arm), name=name).ap
        if (i + 1) % 10 == 0:
            print(f"  rewards {i+1}/{len(names)}", flush=True)
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="training")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cache = FlowCache()
    arms = build_arms()
    labels = [arm_label(a) for a in arms]

    train_names = split_names(args.train_split)
    eval_names = split_names(args.eval_split)
    if args.limit:
        train_names, eval_names = train_names[: args.limit], eval_names[: args.limit]

    for split, names in [(args.train_split, train_names), (args.eval_split, eval_names)]:
        missing = [n for n in names if not cache.has(n)]
        if missing:
            raise SystemExit(
                f"{len(missing)}/{len(names)} {split} images not cached. "
                "Run scripts/01_cache_flows.py first."
            )

    print(f"{len(arms)} arms | train {len(train_names)} | eval {len(eval_names)}")

    # ---- features (image only, no labels) ----
    print("\nextracting features...")
    Xtr_raw = np.array([raw_features(load_sample(n).image) for n in train_names])
    Xev_raw = np.array([raw_features(load_sample(n).image) for n in eval_names])
    scaler = FeatureScaler.fit(Xtr_raw)  # fit on TRAIN only
    Xtr, Xev = scaler.transform(Xtr_raw), scaler.transform(Xev_raw)
    for k, name in enumerate(FEATURE_NAMES):
        print(f"  {name:20s} train mean={Xtr_raw[:,k].mean():.4f} sd={Xtr_raw[:,k].std():.4f}")

    # ---- reward matrices ----
    print("\ncomputing training reward matrix...")
    t0 = time.time()
    gts_tr = {n: load_sample(n).labels for n in train_names}
    Rtr = reward_matrix(cache, train_names, arms, gts_tr)
    print(f"  {time.time()-t0:.0f}s")

    print("computing eval reward matrix (for the oracle reference)...")
    gts_ev = {n: load_sample(n).labels for n in eval_names}
    Rev = reward_matrix(cache, eval_names, arms, gts_ev)

    # ---- train LinUCB ----
    print(f"\ntraining LinUCB: {args.rounds} rounds, alpha={args.alpha}")
    algo = LinUCB(len(arms), scaler.n_features, alpha=args.alpha, seed=args.seed)
    rng = np.random.default_rng(args.seed + 7)
    for _ in range(args.rounds):
        i = int(rng.integers(0, len(train_names)))
        x = Xtr[i]
        a = algo.select(x)
        algo.update(a, x, Rtr[i, a])

    chosen = np.array([algo.greedy_policy(Xev[i]) for i in range(len(eval_names))])
    n_distinct = len(set(chosen.tolist()))
    print(f"  policy selected {n_distinct} distinct arms across {len(eval_names)} eval images")

    # ---- the four numbers ----
    default_idx = labels.index(arm_label(CellposeParams()))
    global_idx = int(np.argmax(Rtr.mean(axis=0)))  # chosen on TRAIN, not eval

    rows = {
        "default": Rev[:, default_idx],
        "global_best": Rev[:, global_idx],
        "contextual": Rev[np.arange(len(eval_names)), chosen],
        "oracle_per_image": Rev.max(axis=1),
    }

    print(f"\n{'strategy':<18} {'AP':>8}  {'95% CI':>18}   detail")
    summary = {}
    for name, vals in rows.items():
        lo, hi = M.bootstrap_ci(vals, seed=args.seed)
        detail = {
            "default": labels[default_idx],
            "global_best": labels[global_idx],
            "contextual": f"{n_distinct} distinct arms",
            "oracle_per_image": "upper bound (uses GT)",
        }[name]
        print(f"{name:<18} {vals.mean():>8.4f}  [{lo:.4f}, {hi:.4f}]   {detail}")
        summary[name] = {
            "ap": float(vals.mean()),
            "ci95": [lo, hi],
            "detail": detail,
        }

    # ---- the honest verdict ----
    gb, cx, orc = rows["global_best"].mean(), rows["contextual"].mean(), rows["oracle_per_image"].mean()
    headroom = orc - gb
    captured = (cx - gb) / headroom if headroom > 1e-9 else 0.0
    print(f"\nper-image adaptation headroom (oracle - global best): {headroom:+.4f}")
    if headroom < 0.005:
        print("  VERDICT: essentially no headroom. Even a perfect per-image chooser could")
        print("  not beat one global configuration on this data. The contextual hypothesis")
        print("  is not supported here, and that is the finding.")
    else:
        print(f"  contextual policy captured {100*captured:.0f}% of the available headroom "
              f"({cx-gb:+.4f} of {headroom:+.4f})")
        lo_g, hi_g = summary["global_best"]["ci95"]
        lo_c, hi_c = summary["contextual"]["ci95"]
        if not (hi_g < lo_c or hi_c < lo_g):
            print("  NOTE: intervals overlap -- with this sample size the gain over the")
            print("  global best is inside the noise. Report it as such.")

    # per-image scores for the chosen policy, so the failure analysis can use them
    scores = [
        M.score_image(gts_ev[n], cache.masks(n, arms[chosen[i]]), name=n)
        for i, n in enumerate(eval_names)
    ]
    agg = save_results(
        f"contextual_{args.eval_split}",
        scores,
        params={"policy": "linucb", "alpha": args.alpha, "rounds": args.rounds},
        extra={
            "strategies": summary,
            "headroom": float(headroom),
            "headroom_captured_frac": float(captured),
            "feature_names": list(FEATURE_NAMES),
            "scaler": scaler.to_dict(),
            "arms": labels,
            "chosen_arm_per_image": {n: labels[chosen[i]] for i, n in enumerate(eval_names)},
        },
    )
    print_summary(f"contextual_{args.eval_split}", agg)

    out = RESULTS / f"contextual_bandit_{args.eval_split}.json"
    out.write_text(json.dumps({"strategies": summary, "headroom": float(headroom)}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
