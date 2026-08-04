#!/usr/bin/env python
"""Precompute and cache the Cellpose-SAM network output for every image.

This is the one expensive step (~9 s per field on Apple Silicon MPS). Everything
downstream -- the parameter sweep, the contextual bandit, the final evaluation --
reads the cache and recomputes masks in ~0.44 s, so this runs once and the
optimizers become practical.

Idempotent: images already cached are skipped, so it can be interrupted and
resumed.

    python scripts/01_cache_flows.py                    # all splits
    python scripts/01_cache_flows.py --splits validation
"""

from __future__ import annotations

import argparse
import time

from nucleiseg.data import load_sample, split_names
from nucleiseg.segmenters import FlowCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "training", "test"],
        help="order matters: validation is needed first for tuning",
    )
    ap.add_argument("--model", default="cpsam_v2")
    ap.add_argument("--augment", action="store_true", help="test-time augmentation")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cache = FlowCache(model_name=args.model, device=args.device, augment=args.augment)
    print(f"model={args.model} augment={args.augment} device={cache.device}")
    print(f"cache -> {cache.cache_dir}")

    for split in args.splits:
        names = split_names(split)
        todo = [n for n in names if not cache.has(n)]
        print(f"\n{split}: {len(names)} images, {len(todo)} to compute")
        t_split = time.time()
        for i, name in enumerate(todo, 1):
            t0 = time.time()
            cache.build(name, load_sample(name).image)
            print(f"  [{i}/{len(todo)}] {name[:26]} {time.time()-t0:.1f}s", flush=True)
        if todo:
            print(f"{split} done in {(time.time()-t_split)/60:.1f} min")

    total = sum(len(split_names(s)) for s in args.splits)
    have = sum(1 for s in args.splits for n in split_names(s) if cache.has(n))
    print(f"\ncached {have}/{total} images")


if __name__ == "__main__":
    main()
