#!/usr/bin/env python
"""Tune Cellpose-SAM post-processing parameters with a bandit, on validation only.

Two things happen here, and the second is what makes the first a result rather
than an assertion:

1. A bandit (UCB1 and Thompson sampling) searches the parameter grid under a
   fixed evaluation budget.
2. The **exhaustive grid** is also evaluated, so we know the true best arm and can
   state whether the bandit found it and at what fraction of the cost. Without
   this the bandit's output is unfalsifiable.

Why a bandit at all: average precision is non-differentiable (thresholding plus
one-to-one matching), so there is no gradient and the choice is between blind
enumeration and adaptive search. The reward is also noisy -- per-image AP spans
roughly 0.70 to 0.87 -- which is exactly the regime bandits are built for.

Tuning uses validation only. The test split is touched once, later, by script 04.

    python scripts/02_bandit_sweep.py --budget 300
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from nucleiseg import metrics as M
from nucleiseg.bandits import ThompsonSampling, UCB1, run_bandit
from nucleiseg.data import load_sample, split_names
from nucleiseg.evaluate import RESULTS
from nucleiseg.grids import GRID, arm_label, build_arms
from nucleiseg.segmenters import FlowCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--budget", type=int, default=300, help="bandit pulls")
    ap.add_argument("--limit", type=int, default=None, help="use first N images only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-exhaustive", action="store_true")
    args = ap.parse_args()

    names = split_names(args.split)
    if args.limit:
        names = names[: args.limit]
    arms = build_arms()
    labels = [arm_label(a) for a in arms]
    cache = FlowCache()

    missing = [n for n in names if not cache.has(n)]
    if missing:
        raise SystemExit(
            f"{len(missing)} images are not cached. Run scripts/01_cache_flows.py first."
        )

    # Ground truth labels are loaded once; the reward function is called
    # thousands of times and must not re-read from disk.
    print(f"loading {len(names)} ground-truth masks...")
    gts = {n: load_sample(n).labels for n in names}

    memo: dict[tuple[int, int], float] = {}

    def reward(arm: int, ctx: int) -> float:
        """AP of configuration `arm` on image `ctx`. Memoised: deterministic."""
        key = (arm, ctx)
        if key not in memo:
            name = names[ctx]
            pred = cache.masks(name, arms[arm])
            memo[key] = M.score_image(gts[name], pred, name=name).ap
        return memo[key]

    print(f"{len(arms)} arms, {len(names)} images, budget {args.budget} pulls")

    results = {}
    for algo_name, algo in [
        ("ucb1", UCB1(len(arms), c=1.0, seed=args.seed)),
        ("thompson", ThompsonSampling(len(arms), seed=args.seed)),
    ]:
        t0 = time.time()
        res = run_bandit(algo, labels, reward, args.budget, len(names), seed=args.seed)
        best = res.best_arm
        print(f"\n[{algo_name}] {time.time()-t0:.1f}s")
        print(f"  best arm: {labels[best]}  mean reward {res.means[best]:.4f}"
              f"  ({res.counts[best]} pulls)")
        for row in res.summary()[:5]:
            print(f"    {row['arm']}  pulls={row['pulls']:3d}  mean={row['mean_reward']:.4f}")
        results[algo_name] = {
            "best_arm": labels[best],
            "best_params": arms[best].__dict__,
            "mean_reward": float(res.means[best]),
            "pulls": int(res.counts[best]),
            "budget": args.budget,
            "arms": res.summary(),
        }

    if not args.skip_exhaustive:
        print(f"\nexhaustive grid: {len(arms)} arms x {len(names)} images "
              f"= {len(arms)*len(names)} evaluations")
        t0 = time.time()
        full = []
        for i, arm in enumerate(arms):
            aps = [reward(i, c) for c in range(len(names))]
            full.append(float(np.mean(aps)))
            print(f"  {labels[i]}  AP={full[-1]:.4f}", flush=True)
        true_best = int(np.argmax(full))
        elapsed = time.time() - t0
        print(f"\nexhaustive took {elapsed/60:.1f} min")
        print(f"true best arm: {labels[true_best]}  AP={full[true_best]:.4f}")
        results["exhaustive"] = {
            "true_best_arm": labels[true_best],
            "true_best_params": arms[true_best].__dict__,
            "true_best_ap": full[true_best],
            "per_arm_ap": dict(zip(labels, full)),
            "n_evaluations": len(arms) * len(names),
        }
        for algo_name in ("ucb1", "thompson"):
            found = results[algo_name]["best_arm"]
            agree = found == labels[true_best]
            gap = full[true_best] - results["exhaustive"]["per_arm_ap"][found]
            frac = args.budget / (len(arms) * len(names))
            print(f"  {algo_name}: {'FOUND the optimum' if agree else 'missed'}"
                  f" using {100*frac:.0f}% of the evaluations"
                  f"{'' if agree else f' (AP gap {gap:.4f})'}")
            results[algo_name]["found_optimum"] = bool(agree)
            results[algo_name]["ap_gap_to_optimum"] = float(gap)
            results[algo_name]["budget_fraction"] = float(frac)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"bandit_sweep_{args.split}.json"
    out.write_text(json.dumps({"grid": GRID, "seed": args.seed, **results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
