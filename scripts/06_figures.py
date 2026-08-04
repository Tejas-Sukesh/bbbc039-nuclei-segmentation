#!/usr/bin/env python
"""Render every figure the writeup cites, from the saved result files.

Reads `results/` rather than recomputing, so a figure and the number quoted
beside it in the README cannot drift apart. Run the evaluation and the failure
analysis first:

    python -m nucleiseg.evaluate --split validation --tag cellpose_default_validation
    python scripts/05_failure_analysis.py --split validation
    python scripts/06_figures.py

Worst-case panels are chosen by sorting the per-image CSV, never by eye, so the
failure section is driven by the measured worst fields rather than by whichever
example made the point most conveniently.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from nucleiseg import failures as F
from nucleiseg import viz
from nucleiseg.data import REPO_ROOT, load_sample, split_names
from nucleiseg.evaluate import RESULTS
from nucleiseg.segmenters import CellposeParams, FlowCache

FIGURES = REPO_ROOT / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--tag", default="cellpose_default_validation")
    ap.add_argument("--failures-tag", default=None)
    ap.add_argument("--worst", type=int, default=4, help="worst-case panels to render")
    args = ap.parse_args()

    ftag = args.failures_tag or f"failures_{args.split}"
    summary = json.loads((RESULTS / f"{args.tag}_summary.json").read_text())
    fsummary = json.loads((RESULTS / f"{ftag}_summary.json").read_text())
    per_image = RESULTS / f"{args.tag}_per_image.csv"
    missed_csv = RESULTS / f"{ftag}_missed_objects.csv"
    fits_csv = RESULTS / f"{ftag}_boundary_fits.csv"

    FIGURES.mkdir(parents=True, exist_ok=True)
    cache = FlowCache()
    params = CellposeParams()
    predict = lambda name: cache.masks(name, params)  # noqa: E731
    made = []

    print("1/6 where the score is lost")
    made.append(viz.ap_by_threshold(
        summary["summary"]["per_threshold_macro"],
        FIGURES / "fig1_where_the_score_is_lost.png",
    ))

    print("2/6 annotation ceiling (whose outline is right)")
    made.append(viz.annotation_ceiling(
        fits_csv,
        FIGURES / "fig2_annotation_ceiling.png",
        summary=fsummary["annotation_ceiling"],
    ))

    print("3/6 what gets missed")
    # Areas of every ground-truth nucleus in the split, as the reference
    # population the missed ones are compared against.
    areas = []
    for name in split_names(args.split):
        labels = load_sample(name).labels
        if labels.max():
            areas.extend(np.bincount(labels.ravel())[1:].tolist())
    made.append(viz.what_gets_missed(
        missed_csv, np.array(areas), FIGURES / "fig3_what_gets_missed.png"
    ))

    print("4/6 merge close-ups")
    with (RESULTS / f"{ftag}_merged_predictions.csv").open() as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: -int(r["n_absorbed"]))
    merges = [
        F.MergedPrediction(
            name=r["name"],
            pred_id=int(r["pred_id"]),
            gt_ids=tuple(int(g) for g in r["gt_ids"].split()),
            gt_areas=tuple(int(a) for a in r["gt_areas"].split()),
            area=int(r["area"]),
            bbox=(int(r["y0"]), int(r["y1"]), int(r["x0"]), int(r["x1"])),
        )
        for r in rows
    ]
    # Show both classes rather than the six worst, which are all one class: the
    # point of the figure is that the two mechanisms look nothing alike.
    by_kind = {k: [m for m in merges if m.kind == k] for k in ("satellite", "comparable")}
    # Three of each, so the two rows of the gallery are the two mechanisms.
    chosen = by_kind["satellite"][:3] + by_kind["comparable"][:3]
    mk = fsummary["merges"]["by_kind"]
    made.append(viz.merge_gallery(
        chosen, predict, FIGURES / "fig4_merges.png", n=6,
        subtitle=f"top row: {mk['satellite']} of {fsummary['merges']['n_merges']} merges"
                 f" are a nucleus + a small punctum.   bottom row: only"
                 f" {mk['comparable']} fuse nuclei of similar size",
    ))

    print("5/6 per-image score spread")
    made.append(viz.score_histogram(per_image, FIGURES / "fig5_per_image_scores.png"))

    print(f"6/6 worst {args.worst} fields")
    made += viz.worst_cases(per_image, predict, args.worst, FIGURES / "worst_cases")

    print(f"\nwrote {len(made)} figures to {FIGURES.relative_to(REPO_ROOT)}/")
    for p in made:
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
