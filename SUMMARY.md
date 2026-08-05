# One-page summary

Full detail in [README.md](README.md). This page is the whole project in about
ten minutes, in plain language.

---

## The task

200 microscope photographs. Each contains roughly 118 nuclei — the blobs at the
centre of cells. A person has drawn a line around every one by hand. Write
software that does the same, and honestly measure how close it got.

## What I did first

Ran **Cellpose-SAM**, a well-known pretrained model: **0.79 out of 1** on the
held-out test images. Also wrote my own segmenter from scratch — find the bright
regions, locate their centres, grow outwards: **0.56**. The pretrained model wins,
which is unsurprising.

That could have been the project. It isn't, because of what the answer key turned
out to be.

## The answer key is a trap

Every pixel in it carries a number. Background is 0, nuclei are 1, 2, or 3. That
is every number in the file — four values, for 190 nuclei.

They are not identity numbers. They are **colours**, assigned so two nuclei that
touch never share one. Like colouring a map so no bordering countries match.

Read the file the obvious way and every touching pair of nuclei melts into a
single blob. Your score goes **up**, because the hardest cases quietly disappear
from the test. I caught it and verified the fix two ways: 23,617 objects
recovered, matching the dataset's own documentation.

## Where the score actually goes

0.79 is not one measurement. It is ten, averaged — each asking how much your
outline must overlap the human's to count as correct, from 50% up to 95%.

| how strict | score |
|---|---|
| loosest (50% overlap) | 0.93 |
| strictest (95%) | 0.31 |

Split the lost points by **cause** rather than by threshold:

| cause | share of the loss |
|---|---|
| failing to find nuclei at all | **37%** |
| finding them, drawing the line slightly wrong | **63%** |

Most of the "error" was never about missing cells.

## What actually fails

**One population: small and faint.** The model finds 98% of bright nuclei and 86%
of dim ones. The typical missed object is 20 pixels; a typical nucleus is 622.

And the metric's "merges" are mostly not merges — of 103 cases where the model
supposedly glued two nuclei together, **60% are a normal nucleus plus a tiny
speck** the annotator marked as its own cell. That is a disagreement about what
counts as a cell, not a failure to separate two of them.

## Twelve things I tried

Adjusting thresholds, size filters, bandit search, averaging over flipped copies,
doubling the resolution, rebalancing local brightness. **All null.** But each
eliminated a mechanism, which is what narrowed the problem down.

Two are worth stating plainly:

- **The bandit lost to brute force.** I built three search algorithms, then
  measured them against simply trying every combination. Brute force won — because
  my own caching had already made the whole grid enumerable in 1.6 minutes. I
  removed the reason for the thing I built.
- **One prediction I registered in advance broke.** I expected doubling the
  resolution to recover small objects and leave merges alone. The opposite
  happened: small objects didn't move at all, and merges dropped by a third.

## The one thing I optimised, measured

**I could not improve accuracy.** The best of twelve attempts is +0.002 against a
noise band of ±0.018.

What I can report is compute: **8.8 seconds → 0.11 seconds per experiment, about
72× faster, with output verified bit-identical.** The model's pipeline splits into
an expensive neural network pass and cheap post-processing, and every tunable
setting only touches the cheap half. So run the network once per image, cache it,
and re-run only the fast part. Trying every setting went from 2.2 hours to 1.6
minutes — which is what made every measurement above affordable.

## What I built and trained

Erase every nucleus the model *did* find, painting it over with plain background.
Now look again. The faint specks it missed used to be competing with 118 bright
nuclei; now they are the brightest things in the picture. Same objects, far easier
problem.

A simple detector circles anything speck-shaped, generously. Then a **classifier I
trained on the 100 held-out training images** decides which circles are real,
judging each on eleven measurements: size, roundness, brightness, brightness
relative to nearby noise, edge sharpness, and whether other nuclei are near.

**Hand-written rules got 28% of those judgements right. The trained model gets
78%.** The clue it leans on second-hardest is signal-to-noise — which is exactly
what every earlier failure had implied, reached independently.

## And the score still went down

F1 0.962 → 0.959.

Not because it finds the wrong things. It finds them in the right places and draws
them slightly too small. To count as correct your outline must overlap the human's
by half; at 44% overlap it is scored as **two mistakes** — the human's nucleus
counts as missed, and yours counts as a false alarm. **Being nearly right is
punished harder than not trying.**

And you cannot fix it by drawing bigger, because of geometry:

| object size | how far off your outline may be |
|---|---|
| 622 pixels (a typical nucleus) | 2.42 pixels |
| 16 pixels (a typical miss) | **0.39 pixels** |

Placing an edge to within four tenths of a pixel, on a faint fuzzy object about
four pixels across, is not achievable — not by the model, and not by the human who
drew the original.

## What it all means

**The difficulty of this metric is set by how big the object is, not by how good
the segmentation is.** Small nuclei are graded roughly forty times more strictly
than large ones, by the same rule.

That is why every attempt at the small ones failed, and why a future one would too.

**The recommendation:** score large nuclei on outline overlap, and count small ones
as simply found-or-not — because at that size, agreement about an outline is not
something you can measure.

## What I got wrong along the way

- **I misread the mask format at first.** Caught it when a field of 190 nuclei
  came back containing three objects.
- **One of my own measurements was biased.** I reported that the model's outlines
  were more accurate than the human's, in 48 of 49 images. A reviewer suggested my
  pixel sampling might be skewed, so I built a test with a known right answer — and
  my measurement picked the wrong outline **9 times out of 9**. I retracted the
  finding, fixed the sampling, and re-ran. What survived is narrower and stated as
  such.

## Known limitations

- The pretrained model was likely trained on some of these images, via an overlap
  with a 2018 competition dataset. So 0.79 is not cleanly held out.
- 50 images per split, so differences under about 0.02 are noise.
- Every headline number carries a confidence interval, and the comparison code
  refuses to state a before/after when two intervals overlap.

---

**In one line:** the score was near its ceiling before I arrived — a pretrained
model put it there. What nobody knew was which of the remaining errors are
fixable, which are disagreements about definitions, and which are geometrically
impossible. I can now say which is which, with a number behind each.
