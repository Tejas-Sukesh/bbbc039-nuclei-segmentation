# Resources to look into

Reading list for the BBBC039 nuclei segmentation take-home. Ordered roughly by
when it becomes useful, not by importance.

Each item is tagged with how much I trust it right now:

- **[verified]** — I fetched it and/or checked it against the data on disk.
- **[unverified]** — I have the citation but have not read the contents.
- **[blocked]** — I tried to retrieve it and could not. Noted so it does not
  get quietly forgotten.

---

## 1. The dataset itself

**BBBC039 landing page** — https://bbbc.broadinstitute.org/BBBC039 **[verified]**

Everything below in this section I confirmed against the 200 images now sitting
in `data/raw/`, so it can be treated as fact rather than documentation:

| Property | Value |
|---|---|
| Images | 200 fields of view, 16-bit TIFF, 520 × 696 |
| Actual intensity range | ~120 (camera floor) to 4095 — **not** 0–65535 |
| Masks | 200 RGBA PNGs, `uint8` |
| Total nuclei | 23,617 (dataset page says "~23,000") |
| Nuclei per field | mean 118, min **0**, max 231 |
| Splits | training 100 / validation 50 / test 50 |
| Empty fields | 2 in training, 1 in validation, 0 in test |
| Sample | U2OS cells, Hoechst stain |

Download URLs (used by `scripts/download_data.sh`):
`https://data.broadinstitute.org/bbbc/BBBC039/{images,masks,metadata}.zip`
(78 MB / 2.8 MB / 18 KB).

**The mask encoding is the one real trap in this dataset. [verified]** The page
says only "if two nuclei touch, they are labeled with a different color," which
undersells the problem. What is actually on disk: an RGBA PNG where G and B are
all-zero, A is all-255, and the **red channel holds a 3-color graph coloring,
not instance IDs**. A field with 190 nuclei has exactly four distinct pixel
values (0 for background, 1–3 for nuclei). Loading the red channel as a label
image collapses the field into ≤3 connected blobs and destroys the instance
structure — and since touching nuclei are the whole difficulty, it destroys
exactly what we are trying to measure, while still producing plausible-looking
output. Instances are recovered by running connected components *within each
color* and concatenating; this is valid because same-color nuclei are guaranteed
non-adjacent by construction. Implemented and cross-checked against the
published nucleus count in [`src/nucleiseg/data.py`](src/nucleiseg/data.py).

Also worth knowing:

- `metadata/segmentation_cp3.cppipe` **[verified present]** is a CellProfiler 3
  baseline pipeline shipped with the dataset. Worth reading as a spec for what
  a competent classical pipeline does stage by stage, even without running
  CellProfiler. This is the closest thing to an official baseline.
- `metadata/filenames_and_plates.csv` maps each field to its BBBC022 plate.
  Useful because the published splits are plate-grouped — a random re-split
  would leak plate-level illumination and confluence effects across train/test.
- BBBC039 is sampled from **BBBC022** (a Cell Painting chemical screen), so the
  other fluorescence channels exist upstream if multi-channel context ever
  helps. A small fraction overlaps **BBBC038** (Data Science Bowl 2018).

## 2. Evaluation methodology

This is the part worth getting right before writing any segmenter, since the
metric determines which failures are even visible.

**Caicedo et al., "Evaluation of Deep Learning Strategies for Nucleus
Segmentation in Fluorescence Images," Cytometry Part A 95(9):952–965, 2019.**
https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.23863 — **[blocked]**

This is *the* companion paper for BBBC039 and the source of the reference
numbers to compare against. Publisher returns 403, and the bioRxiv preprint
(10.1101/335216) and its PDF also 403'd. **To do:** retrieve via institutional
access, or pull the numbers out of the authors' code and result files at
https://github.com/carpenterlab/2019_caicedo_cytometryA **[unverified]** — that
repo is public and its evaluation notebooks are the authoritative definition of
the metric for this dataset. Until then I have no published number to anchor
"is 0.7 good?", which is a real gap in the writeup.

**Caicedo et al., "Nucleus segmentation across imaging experiments: the 2018
Data Science Bowl," Nature Methods 16:1247–1253, 2019.**
https://pmc.ncbi.nlm.nih.gov/articles/PMC6919559/ — **[verified, open access]**

Different (broader) dataset, but the right reference for metric design. Useful
specifics I confirmed in the text:

- Accuracy is reported as **F1 at multiple IoU thresholds**, with the
  area under the F1-vs-IoU curve tracking the overall competition score.
- They single out **IoU 0.7** as an interpretable operating point: objects must
  overlap by ≥70% of their area to count as a true positive.
- The reference baseline was **CellProfiler configured by an expert analyst**;
  top deep-learning entries beat it "by a large margin." Best accuracy reached
  0.90 on the largest image group but only 0.55 on the smallest — i.e.
  performance is strongly image-type dependent, which is a caution against
  reading one aggregate number as skill.

**Data Science Bowl 2018 mean-AP metric** — https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation **[unverified]**

The AP averaged over IoU thresholds 0.50:0.05:0.95 convention. One thing to be
careful about and state explicitly in the writeup: the Kaggle "AP" is
`TP / (TP + FP + FN)` at each threshold — a Jaccard-style ratio over *objects* —
**not** the area-under-the-precision-recall-curve AP from object detection.
The two are different numbers with the same name.

**Matching predictions to ground truth.** Greedy matching by descending IoU is
the DSB2018 convention; optimal one-to-one assignment via
`scipy.optimize.linear_sum_assignment` can differ in dense clumps. Worth
implementing both and reporting whether they disagree. See
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html **[unverified]**

**Also read:** Maier-Hein et al., "Metrics reloaded: recommendations for image
analysis validation," Nature Methods 21:195–212, 2024
(https://www.nature.com/articles/s41592-023-02151-z) **[unverified]** — a
systematic treatment of how to pick a segmentation metric and which ones
mislead. Directly relevant to justifying the metric choice rather than
defaulting to it.

## 3. Classical baseline (the plan of record)

Every stage is inspectable, which is what makes a failure analysis possible.
Standard references:

- **scikit-image watershed segmentation example** —
  https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html **[unverified]**
  The distance-transform → local-maxima seeds → watershed recipe.
- **`skimage.segmentation.watershed`, `skimage.filters.threshold_otsu` /
  `threshold_local`, `skimage.feature.peak_local_max`,
  `skimage.morphology.h_maxima`, `scipy.ndimage.distance_transform_edt`** —
  API docs at https://scikit-image.org/docs/stable/api/api.html **[verified installed]**
  (scikit-image 0.26.0 in `.venv`).
- **Otsu (1979), "A threshold selection method from gray-level histograms."**
  Worth actually understanding rather than calling: it maximizes between-class
  variance assuming a **bimodal** histogram. That assumption is what breaks on
  the near-empty fields and on fields with bright debris — a predictable,
  explainable failure, which makes it a good thing to have in the writeup.
- **h-maxima / extended-maxima suppression** for over-seeding. The seed stage
  is where touching nuclei are won or lost, so seed depth / `min_distance` is
  the most consequential single parameter in the pipeline and the natural
  candidate for the required "optimize one thing deliberately."

Known failure modes to look for (and the reason a classical baseline is
interesting rather than just weak): touching/clumped nuclei merging under
watershed, over-segmentation of large or irregular nuclei, out-of-focus fields,
bright debris and saturated artifacts dragging a global threshold, mitotic and
apoptotic nuclei with atypical shape and intensity, and nuclei clipped at the
image border.

## 4. Deep learning, if there is time

Not the plan of record — the ask rewards failure analysis over leaderboard
position, and a baseline whose every stage is legible serves that better. Kept
here as the honest comparison point and the "what I'd do next" section.

- **U-Net** — Ronneberger et al. 2015, https://arxiv.org/abs/1505.04597
  **[unverified]**. The architecture Caicedo et al. evaluate on this exact
  dataset. Note their setup predicts **three classes** (background / nucleus
  interior / nucleus boundary) rather than binary foreground, precisely so that
  touching nuclei separate — the learned analogue of the watershed seed problem.
- **StarDist** — Schmidt et al. 2018, https://arxiv.org/abs/1806.03535, code at
  https://github.com/stardist/stardist **[verified installable]**: PyPI 0.9.2
  ships `macosx_12_0_arm64` wheels for cp310–cp313, so it installs on Apple
  Silicon. Caveat: it pulls in `csbdeep` → **TensorFlow**, which is the real
  friction on this machine; budget setup time. Star-convex polygon
  representation is a strong fit for roundish nuclei and it is the best
  accuracy-per-setup-hour option here.
- **Cellpose** — Stringer et al., Nature Methods 2021,
  https://github.com/MouseLand/cellpose **[verified installable]**: PyPI 4.2.1.1
  is a pure-Python wheel (`py3-none-any`) on PyTorch, so Apple Silicon is fine
  via the **MPS** backend, no CUDA needed. Pretrained `nuclei` model means
  **zero-shot inference with no training** — by far the cheapest way to get a
  strong number for comparison.
- **Mask R-CNN** — He et al. 2017, https://arxiv.org/abs/1703.06870
  **[unverified]**. Also evaluated by Caicedo et al. Realistically too much
  setup for the time budget; listed for completeness.

Practical note: no CUDA on this machine. Anything torch-based should use
`torch.device("mps")`; TensorFlow-based work needs the arm64 build.

## 5. Tooling

All **[verified installed]** in `.venv` (Python 3.11.15): numpy 2.x, scipy
1.17.1, scikit-image 0.26.0, tifffile 2026.3.3, imageio, matplotlib, pandas,
tqdm.

- `tifffile` for the 16-bit TIFFs — https://github.com/cgohlke/tifffile
- `imageio.v3` for the RGBA mask PNGs (`iio.imread` preserves all 4 channels;
  needed to get at the red channel)
- `scipy.ndimage.label` for the per-color connected components in mask decoding
- Use a **random-permutation colormap** for label images. A sequential colormap
  makes adjacent instance IDs nearly indistinguishable, which will hide exactly
  the split/merge errors we are hunting.

---

## Open questions to resolve

1. **No reference number yet.** Caicedo et al. 2019 is paywalled; without it
   there is no published anchor for what score is good on BBBC039. Next move is
   the authors' GitHub result files.
2. **How should empty fields enter the aggregate?** Three fields have zero GT
   nuclei. A field with no GT can be either a free perfect score or an
   all-false-positive zero depending on convention, and with 50 test images one
   field is 2% of the mean. Needs an explicit, documented decision. (Test has
   none, which limits the damage, but validation has one and that is the tuning
   set.)
3. **Aggregate by mean-over-images or pooled-over-objects?** Pooling weights
   dense fields more heavily and would hide the sparse-field failures entirely.
   Leaning mean-over-images, stated explicitly.
4. **Smallest GT nuclei are ~13 px.** Any `min_area` post-filter has to stay
   below that or it silently deletes real objects; check the GT size
   distribution before setting it.
