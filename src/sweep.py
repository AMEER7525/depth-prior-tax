"""Grid expansion with degeneracy pruning (Axis A x B x C x lambda).

A corrupted depth map can reach a 3DGS run through exactly two paths:

    P1  initialization -- active iff init == "depth" (backprojection consumes
        METRIC depth, so an affine bias genuinely displaces the point cloud)
    P2  the depth loss -- active iff depth_lambda > 0

If neither path is active the depth settings are inert and every distinct depth
config collapses to one run. If only P2 is active, what survives depends on the
loss: Pearson is blind to an affine bias, SSI turns it into a pure gain on
lambda, and only the absolute loss registers it as error. Those facts are
pinned by tests/test_depth_semantics.py.

Pruning the cells this implies is not an optimization -- an unpruned grid
reports "no effect" for cells that were incapable of showing one, which would
read as evidence against the hypothesis.
"""
from __future__ import annotations

import itertools

# Losses that cannot see a global affine bias as geometry error.
_AFFINE_INVARIANT = {"ssi", "pearson"}
# Canonical value used when an axis is inert, so dedupe is deterministic.
_CANONICAL_KIND = "ssi"


def _is_affine_only(cell):
    """Corruption is a pure affine bias: scale/shift set, no additive noise."""
    biased = cell.get("scale", 1.0) != 1.0 or cell.get("shift", 0.0) != 0.0
    return biased and cell.get("sigma_rel", 0.0) == 0.0


def _is_clean(cell):
    return (cell.get("sigma_rel", 0.0) == 0.0
            and cell.get("scale", 1.0) == 1.0
            and cell.get("shift", 0.0) == 0.0)


def degenerate_reason(cell):
    """Why this cell cannot produce an independent result, or None if it can."""
    init_uses_depth = cell.get("init") == "depth"
    loss_active = cell.get("depth_lambda", 0.0) > 0
    kind = cell.get("depth_kind", _CANONICAL_KIND)

    # The depth prior has no path into the run at all.
    if not init_uses_depth and not loss_active:
        if not _is_clean(cell) or kind != _CANONICAL_KIND:
            return ("depth prior is inert (init != depth, lambda == 0): "
                    "duplicates the clean no-depth run")
        return None

    # lambda == 0 with depth init: the loss kind is never consulted.
    if init_uses_depth and not loss_active and kind != _CANONICAL_KIND:
        return "depth_kind is unused at lambda == 0: duplicates depth_kind=ssi"

    # Loss-only path, affine-only corruption: what the loss can actually see.
    if loss_active and not init_uses_depth and _is_affine_only(cell):
        if kind == "pearson":
            return ("pearson is invariant to affine bias: bit-identical to the "
                    "clean run")
        if kind == "ssi":
            scale = cell.get("scale", 1.0)
            return (f"ssi rescales affine bias into lambda: aliases "
                    f"(clean, lambda={cell['depth_lambda'] * scale:g})")
    return None


def expand(grid, keep_degenerate=False):
    """Expand a sweep config into (cells, pruned) lists of run dicts.

    `pruned` entries carry a "_reason" key so the skip is auditable rather than
    silent -- the report needs to say which cells were dropped and why.
    """
    axes = grid["axes"]
    keys = list(axes)
    scenes = grid.get("scenes") or [None]

    cells, pruned, seen = [], [], set()
    for combo in itertools.product(*(axes[k] for k in keys)):
        base = dict(zip(keys, combo))
        for scene in scenes:
            cell = {**base, "scene": scene}
            reason = degenerate_reason(cell)
            if reason and not keep_degenerate:
                pruned.append({**cell, "_reason": reason})
                continue
            key = run_id(cell)
            if key in seen:          # distinct configs, identical run
                pruned.append({**cell, "_reason": f"duplicate run id {key}"})
                continue
            seen.add(key)
            cells.append(cell)
    return cells, pruned


def run_id(cell):
    """Stable, filesystem-safe identifier. Also the resume key."""
    def fmt(v):
        return str(v).replace(".", "p").replace("-", "m")

    parts = [str(cell.get("scene", "scene"))]
    for k in ("init", "n_views", "depth_kind", "depth_lambda",
              "sigma_rel", "scale", "shift"):
        if k in cell:
            parts.append(f"{k}-{fmt(cell[k])}")
    return "__".join(parts)
