"""Loading BBBC039 images and ground-truth instance masks.

The only subtle part of this dataset is the mask encoding, documented in
`decode_mask` below. Everything else is a plain TIFF read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
IMAGES, MASKS, METADATA = RAW / "images", RAW / "masks", RAW / "metadata"

SPLITS = ("training", "validation", "test")


@dataclass(frozen=True)
class Sample:
    """One field of view: the raw image and its ground-truth instance labels."""

    name: str  # stem, no extension
    image: np.ndarray  # (520, 696) uint16, raw camera counts
    labels: np.ndarray  # (520, 696) int32, 0=background, 1..N=nucleus instances

    @property
    def n_nuclei(self) -> int:
        return int(self.labels.max())


def decode_mask(path: str | Path) -> np.ndarray:
    """Recover per-nucleus instance labels from a BBBC039 mask PNG.

    The masks are NOT instance-ID images, which is the trap this dataset sets.
    Each mask is an RGBA PNG in which only the red channel carries signal, and
    that channel holds a *graph coloring* rather than object IDs: background is
    0 and every nucleus gets a color in 1..3, chosen only so that two nuclei
    that touch never share a color. So a field with 190 nuclei still has just
    four distinct pixel values.

    Reading the red channel as labels therefore collapses the whole image into
    <=3 giant "objects" and silently destroys the instance structure -- and,
    because touching nuclei are exactly the hard cases, it destroys precisely
    what we want to measure. Instances are recovered by running connected
    components *within each color* and concatenating the results, which is
    sound because same-color nuclei are guaranteed non-adjacent by
    construction.

    Verified empirically over all 200 masks: 23,617 nuclei recovered, matching
    the ~23,000 the dataset page reports. Channels G/B are all-zero and A is
    all-255 in every mask.
    """
    raw = iio.imread(path)
    coloring = raw[..., 0] if raw.ndim == 3 else raw

    labels = np.zeros(coloring.shape, dtype=np.int32)
    offset = 0
    for color in range(1, int(coloring.max()) + 1):
        cc, n = ndi.label(coloring == color)
        # Shift this color's IDs past the ones already assigned.
        labels[cc > 0] = cc[cc > 0] + offset
        offset += n
    return labels


def split_names(split: str) -> list[str]:
    """Image stems belonging to a published split ('training'/'validation'/'test').

    Use these rather than re-splitting, so the numbers stay comparable to
    published work on this dataset.

    Note what the split does *not* do: it is field-level, not plate-level. All
    20 BBBC022 plates represented in this dataset appear in all three splits
    (checked against metadata/filenames_and_plates.csv), so no imaging batch is
    held out. Test performance therefore measures generalization across fields
    of the same experiment, not across plates, microscopes, or staining runs.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    listing = (METADATA / f"{split}.txt").read_text().split()
    return [Path(line).stem for line in listing if line.strip()]


def load_sample(name: str) -> Sample:
    """Load one field of view by stem."""
    return Sample(
        name=name,
        image=tifffile.imread(IMAGES / f"{name}.tif"),
        labels=decode_mask(MASKS / f"{name}.png"),
    )


def load_split(split: str):
    """Lazily yield every Sample in a split, in the order the metadata lists them."""
    for name in split_names(split):
        yield load_sample(name)


if __name__ == "__main__":
    # Sanity check: python -m nucleiseg.data
    for split in SPLITS:
        counts = [s.n_nuclei for s in load_split(split)]
        empty = sum(c == 0 for c in counts)
        print(
            f"{split:11s} images={len(counts):3d} nuclei={sum(counts):6d} "
            f"mean={np.mean(counts):6.1f} min={min(counts):3d} max={max(counts):3d} "
            f"empty_fields={empty}"
        )
