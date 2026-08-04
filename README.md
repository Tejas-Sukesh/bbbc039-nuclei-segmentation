# Nuclei instance segmentation on BBBC039

Take-home for the Neo Scholar program. Segments individual nuclei from
fluorescence microscopy images and scores itself against hand-annotated ground
truth.

> **Status: scaffold.** Data loading and ground-truth decoding are implemented
> and validated against the dataset's published nucleus count. The segmenter,
> metrics, and evaluation harness are stubbed with their design decisions
> written down. Results sections below are placeholders and are marked as such.

## Quickstart

From a clean clone:

```bash
git clone https://github.com/Tejas-Sukesh/bbbc039-nuclei-segmentation.git
cd bbbc039-nuclei-segmentation
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
bash scripts/download_data.sh          # ~80 MB from the Broad Institute
python -m nucleiseg.data                # sanity check: prints per-split stats
```

That last command should print:

```
training    images=100 nuclei= 12001 mean= 120.0 min=  0 max=194 empty_fields=2
validation  images= 50 nuclei=  5896 mean= 117.9 min=  0 max=231 empty_fields=1
test        images= 50 nuclei=  5720 mean= 114.4 min=  7 max=202 empty_fields=0
```

If the nuclei counts differ, the mask decoding is wrong — see below.

## The dataset

[BBBC039](https://bbbc.broadinstitute.org/BBBC039): 200 fields of U2OS cells,
Hoechst-stained, from the Broad Bioimage Benchmark Collection. 16-bit TIFF at
520 × 696. 23,617 hand-annotated nuclei, ~118 per field. The published
train/validation/test split of 100/50/50 is grouped by plate, so I use it as
given rather than re-splitting — a random split would leak plate-level
illumination and confluence effects across the boundary.

Two things about the raw data that affect the pipeline: intensities occupy
roughly 120–4095 rather than the full 16-bit range (120 is the camera floor, not
black), and three fields contain **zero** nuclei, which makes them a division-by-
zero hazard in any per-image metric.

### The ground truth is not what it looks like

The masks are the one genuinely tricky part of this dataset, and getting them
wrong produces plausible-looking output while silently destroying the thing
being measured.

They are RGBA PNGs. Green and blue are all-zero, alpha is all-255, and the red
channel holds **a 3-color graph coloring rather than instance IDs** — background
is 0 and every nucleus gets a color in 1–3, assigned only so that two nuclei
that touch never share a color. So a field containing 190 nuclei has exactly
four distinct pixel values.

Reading that channel as a label image collapses the whole field into at most
three enormous connected blobs. Worse, the nuclei it fuses are specifically the
*touching* ones — which are exactly the hard cases that any instance
segmentation metric is meant to test. The bug would inflate scores and hide the
principal failure mode.

Recovering instances means running connected components **within each color
separately** and concatenating the results. That is valid precisely because the
coloring guarantees same-color nuclei are never adjacent. Implemented in
[`decode_mask`](src/nucleiseg/data.py); I checked it by counting recovered
objects across all 200 masks and getting 23,617, which matches the "~23,000"
the dataset page reports.

## Approach

A classical, fully inspectable pipeline rather than a neural network — chosen
because the brief weights failure analysis above raw score, and a pipeline whose
every stage can be visualized lets me attribute an error to a *stage* instead of
to a black box.

1. **Normalize** — per-image percentile rescaling, since plate-to-plate
   illumination varies and the intensity floor is nonzero.
2. **Threshold** — Otsu for foreground, compared against adaptive/local
   thresholding.
3. **Seed** — Euclidean distance transform, then local maxima as one seed per
   nucleus. This stage decides whether touching nuclei separate.
4. **Watershed** — flood from the seeds, constrained to the foreground mask.
5. **Post-filter** — drop objects below a minimum area (kept low: the smallest
   ground-truth nucleus is ~13 px).

### Metrics

Average precision over IoU thresholds 0.50:0.05:0.95 (the Data Science Bowl 2018
convention) as the headline number, F1 at IoU 0.5 as the number that is legible
to a biologist counting cells, and **split/merge counts tracked separately**.
That last one matters: a pipeline that over-segments and one that
under-segments can post an identical AP while failing in opposite directions,
and the aggregate alone cannot tell them apart.

One naming caveat worth stating, since the two are easy to conflate: the Kaggle
"AP" used here is `TP / (TP + FP + FN)` at each threshold — a Jaccard-style
ratio over *objects* — not the area under a precision-recall curve.

Aggregation is mean-over-images, not pooled-over-objects, so that dense fields
do not dominate and sparse-field failures stay visible.

## Results

*Placeholder — not yet measured.*

| Method | AP@[.5:.95] | F1@0.5 | Splits | Merges |
|---|---|---|---|---|
| Otsu + watershed (baseline) | TBD | TBD | TBD | TBD |
| + optimized seeding | TBD | TBD | TBD | TBD |

### The one thing optimized deliberately

*Placeholder.* Planned axis: watershed **seed** parameters (h-maxima suppression
depth and minimum seed separation), tuned on validation only and reported once
on test. Chosen because seeding is the stage that determines whether touching
nuclei are separated, making it the highest-leverage parameter in the pipeline —
the before/after should move the merge count directly.

## Where it fails

*Placeholder — to be written from measured per-image results, not from
speculation.* Candidate mechanisms to test, each with a specific reason to
expect it:

- **Clumped/touching nuclei** — the distance transform of a fused clump may have
  a single maximum, so watershed receives one seed and returns one object.
- **Bright debris or saturated artifacts** — Otsu maximizes between-class
  variance assuming a *bimodal* histogram; one very bright blob shifts the
  threshold up and dims real nuclei out of the foreground.
- **Near-empty and empty fields** — with no true bimodality, Otsu must still
  return a threshold, so it splits noise and manufactures false positives.
- **Out-of-focus fields** — blurred edges flatten the intensity gradient, so the
  foreground boundary drifts and IoU degrades even when detection succeeds.
- **Mitotic and apoptotic nuclei** — atypical shape and brightness break both
  the roundness assumption behind distance-transform seeding and the area filter.
- **Border-clipped nuclei** — truncated objects have small area and off-center
  distance maxima.
- **Very small nuclei** — anything near the ~13 px floor competes directly with
  the `min_area` filter meant to suppress noise.

## What I'd try next

*Placeholder.* See [RESOURCES.md](RESOURCES.md) §4 for the deep-learning options
and their setup cost on this machine.

## Repo layout

```
src/nucleiseg/
  data.py       ground-truth decoding + split loading   [implemented]
  metrics.py    IoU matching, AP sweep, split/merge     [stub]
  baseline.py   normalize -> threshold -> seed -> watershed [stub]
  evaluate.py   harness + parameter sweep               [stub]
  viz.py        overlays, error panels, worst cases      [stub]
scripts/download_data.sh
RESOURCES.md    annotated reading list, verified vs. not
```

The stubs carry the design decisions and the reasoning behind them in their
docstrings, so the contract is settled before the implementation.

## Notes

`data/` is gitignored; run `scripts/download_data.sh` to populate it. No GPU
required — the baseline is CPU-only. Developed on Apple Silicon (no CUDA), which
constrains the deep-learning options discussed in RESOURCES.md.
