"""The parameter search space, defined once so every script searches the same one.

Kept in the package rather than in a script because the sweep and the contextual
bandit must enumerate identical arms -- otherwise their results are not
comparable, and the comparison is the point.

Two arm families live here, and the second exists because of what the first
measured.

CHEAP_GRID -- post-processing only
----------------------------------
Parameters recomputable from cached Cellpose flows in ~0.44 s rather than
requiring a ~9 s network pass (see `segmenters.FlowCache`).

**Measured result: the Cellpose defaults are already optimal in this space, and
the hypothesis that motivated the search was wrong.** The reasoning had been that
a measured -5.1% count bias meant too few objects, so lowering
`cellprob_threshold` -- documented to find "more and larger masks" -- should
recover them. Sweeping it in *both* directions on validation shows a clean peak
exactly at the default:

    cellprob_threshold   AP      count bias   splits  merges
              -0.5     0.7930      -4.9%         3      26
              +0.0     0.8028      -5.1%         3      25   <- default, best
              +0.5     0.7838      -5.9%         3      24
              +1.0     0.7514      -6.4%         3      24
              +2.0     0.6695      -7.5%         3      18

Two things to read off this. First, the threshold moves AP a lot but moves the
count bias barely at all, so it is not the mechanism behind the under-counting.
Second, the errors are overwhelmingly **merges** (25) rather than splits (3):
touching nuclei whose flow fields converge on a single centre. No post-processing
threshold can separate them, because by the time the flows are computed the two
nuclei have already become one basin.

Lowering `min_size` below the default also failed to help, despite 96 of the
5,896 validation nuclei being smaller than 15 px -- the noise objects it admits
outweigh the real ones it recovers.

So the -5.1% bias lives in the **learned representation, not the
post-processing**, and closing it requires changing what the network sees or how
many networks vote -- which is what MODEL_ARMS is for. This is the same shape of
conclusion as the classical pipeline's ceiling: the information was destroyed
upstream of the knob being turned.

MODEL_ARMS -- representation-level
----------------------------------
Arms that differ in the network pass itself, so their errors are genuinely
decorrelated rather than being 24 near-identical post-processing variants. This
also gives a contextual bandit something to work with: two backbones can plausibly
disagree about which is better on a *sparse* field versus a *crowded* one, whereas
a threshold sweep mostly just shifts every image the same way.

Each arm needs its own flow cache, so adding one costs a network pass over the
split.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .segmenters import CellposeParams

# --------------------------------------------------------------------------- #
# Cheap: post-processing over one cached flow field
# --------------------------------------------------------------------------- #

CHEAP_GRID: dict[str, list] = {
    # Swept in both directions after the one-sided search proved uninformative.
    "cellprob_threshold": [-0.5, 0.0, 0.5],
    "min_size": [5, 15, 30],
    "flow_threshold": [0.4, 0.5],
}

# Retained under the old name so earlier result files remain interpretable.
GRID = CHEAP_GRID


def build_arms(grid: dict[str, list] | None = None) -> list[CellposeParams]:
    """Cartesian product of the cheap grid, as concrete parameter objects."""
    grid = grid or CHEAP_GRID
    keys = list(grid)
    return [
        CellposeParams(**dict(zip(keys, values)))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def arm_label(p: CellposeParams) -> str:
    """Short stable label, used as a dict key in saved results."""
    return f"cp={p.cellprob_threshold:+.1f},ms={p.min_size:>2d},ft={p.flow_threshold:.1f}"


DEFAULT_LABEL = arm_label(CellposeParams())


# --------------------------------------------------------------------------- #
# Representation-level: different network passes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelArm:
    """One (network configuration, post-processing) pair.

    `model_name` and `augment` determine which flow cache is read, so changing
    either requires a cached network pass over the split. `params` is then free.
    """

    model_name: str
    augment: bool = False
    params: CellposeParams = CellposeParams()

    @property
    def label(self) -> str:
        tag = f"{self.model_name}{'+tta' if self.augment else ''}"
        return f"{tag}|{arm_label(self.params)}"

    def cache(self, device: str | None = None):
        from .segmenters import FlowCache

        return FlowCache(model_name=self.model_name, augment=self.augment, device=device)


# `cpsam_v2` is the paper's current model and the default. `cpsam` is the
# original Cellpose-SAM release; `cpdino` swaps the SAM backbone for a DINOv3
# one, so its failure modes should be the least correlated with the others --
# which is exactly what makes an ensemble or a per-image choice worth testing.
# `+tta` averages over flipped tiles, which should reduce variance in the flow
# field and is the most direct shot at the merge problem.
MODEL_ARMS: list[ModelArm] = [
    ModelArm("cpsam_v2"),
    ModelArm("cpsam_v2", augment=True),
    ModelArm("cpdino"),
]


def model_arm_labels() -> list[str]:
    return [a.label for a in MODEL_ARMS]
