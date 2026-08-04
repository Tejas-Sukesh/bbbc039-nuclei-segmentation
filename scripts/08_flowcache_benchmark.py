#!/usr/bin/env python
"""Measure the speedup FlowCache buys, and prove it changes no output.

This is the project's deliberate optimization, so its before/after has to be
measured rather than asserted, and the equivalence claim has to be checked rather
than argued. A speedup that quietly perturbs the masks is not a speedup.

before : `model.eval(image)` -- the network forward pass plus mask construction,
         every time a parameter changes.
after  : `dynamics.resize_and_compute_masks` over the cached flow field, with the
         network run once per image ever.

The equivalence is exact, not approximate, and it has to be: the cached arrays
are stored as float16 to halve the footprint, which is a lossy cast. The check
below is `np.array_equal` on the resulting label images, which is the property
that actually matters -- identical instances, not identical intermediates.

    python scripts/08_flowcache_benchmark.py --n 3
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from nucleiseg.data import load_sample, split_names
from nucleiseg.evaluate import RESULTS
from nucleiseg.segmenters import CellposeParams, FlowCache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="images to time")
    ap.add_argument("--repeats", type=int, default=3, help="cached-path timings per image")
    args = ap.parse_args()

    names = split_names("validation")[: args.n]
    cache = FlowCache()
    params = CellposeParams()

    from cellpose import dynamics

    rows = []
    for name in names:
        image = load_sample(name).image

        # before: full network pass, the cost paid per parameter trial without a cache
        t0 = time.perf_counter()
        _, flows, _ = cache.model.eval(image, augment=False)
        out = dynamics.resize_and_compute_masks(
            flows[1], flows[2],
            niter=params.niter,
            cellprob_threshold=params.cellprob_threshold,
            flow_threshold=params.flow_threshold,
            min_size=params.min_size,
            max_size_fraction=params.max_size_fraction,
            device=cache.device,
        )
        uncached_s = time.perf_counter() - t0
        direct = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.int32)

        # after: post-processing over the cached float16 flow field
        cached_times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            via_cache = cache.masks(name, params)
            cached_times.append(time.perf_counter() - t0)
        cached_s = float(np.median(cached_times))

        identical = bool(np.array_equal(direct, via_cache))
        rows.append({
            "name": name,
            "uncached_s": uncached_s,
            "cached_s": cached_s,
            "speedup": uncached_s / cached_s,
            "n_objects": int(direct.max()),
            "identical": identical,
        })
        print(f"  {name[:26]}  {uncached_s:6.2f}s -> {cached_s:5.3f}s"
              f"  ({uncached_s/cached_s:5.1f}x)  objects={direct.max():3d}"
              f"  identical={identical}", flush=True)

    uncached = float(np.mean([r["uncached_s"] for r in rows]))
    cached = float(np.mean([r["cached_s"] for r in rows]))
    all_identical = all(r["identical"] for r in rows)

    print(f"\n=== FlowCache: the deliberate optimization ===")
    print(f"  before (network pass per trial) : {uncached:.2f} s")
    print(f"  after  (cached flows per trial) : {cached:.3f} s")
    print(f"  speedup                         : {uncached/cached:.1f}x")
    print(f"  label images bit-identical      : {all_identical}"
          f"   ({len(rows)}/{len(rows)} images)" if all_identical
          else f"  label images bit-identical      : FALSE -- the cache is not sound")
    grid = 18 * 50
    print(f"\n  what it buys: the 18-arm grid over 50 validation images is {grid}"
          f" evaluations,\n  {grid*uncached/3600:.1f} h uncached against"
          f" {grid*cached/60:.1f} min cached.")

    out_path = RESULTS / "flowcache_benchmark.json"
    out_path.write_text(json.dumps({
        "device": str(cache.device),
        "model": cache.model_name,
        "n_images": len(rows),
        "repeats_per_image": args.repeats,
        "uncached_mean_s": uncached,
        "cached_mean_s": cached,
        "speedup": uncached / cached,
        "all_label_images_identical": all_identical,
        "per_image": rows,
    }, indent=2, default=float))
    print(f"\nwrote {out_path.name}")
    if not all_identical:
        raise SystemExit("cache does not reproduce the direct path -- do not trust results")


if __name__ == "__main__":
    main()
