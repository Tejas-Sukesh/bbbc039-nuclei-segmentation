"""The parameter search space, defined once so every script searches the same one.

Kept in the package rather than in a script because both the sweep and the
contextual bandit must enumerate identical arms -- otherwise their results are
not comparable, and the comparison is the whole point.

Only *cheap* parameters appear here: those recomputable from cached Cellpose
flows in ~0.44 s rather than requiring a ~9 s network pass. See
`segmenters.FlowCache` for that split. Conveniently, both mechanisms behind the
measured -5.1% count bias are cheap:

* `min_size` -- Cellpose defaults to 15, but 96 of the 5,896 validation nuclei
  are smaller than that, so the default discards real objects outright.
* `cellprob_threshold` -- documented to find more and larger masks as it
  decreases, which is the direction the measured under-counting calls for.

`flow_threshold` is included as a control: it gates masks whose flows are
inconsistent, so it trades false positives against recall by a different
mechanism, and a sweep should show whether it matters here at all.
"""

from __future__ import annotations

import itertools

from .segmenters import CellposeParams

GRID: dict[str, list] = {
    "cellprob_threshold": [-2.0, -1.0, -0.5, 0.0],
    "min_size": [5, 10, 15],
    "flow_threshold": [0.4, 0.5],
}


def build_arms(grid: dict[str, list] | None = None) -> list[CellposeParams]:
    """Cartesian product of the grid, as concrete parameter objects."""
    grid = grid or GRID
    keys = list(grid)
    return [
        CellposeParams(**dict(zip(keys, values)))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def arm_label(p: CellposeParams) -> str:
    """Short stable label, used as a dict key in saved results."""
    return f"cp={p.cellprob_threshold:+.1f},ms={p.min_size:>2d},ft={p.flow_threshold:.1f}"


DEFAULT_LABEL = arm_label(CellposeParams())
