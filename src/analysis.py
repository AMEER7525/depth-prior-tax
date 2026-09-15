"""Tables and figures for the appearance-vs-geometry study.

Everything is computed from results.csv (scripts/evaluate.py):

  load_results      parse, fill defaults, derive the PATH the prior took into
                    the run, the CORRUPTION arm, and deltas vs vanilla 3DGS
  divergences       cells whose appearance and geometry moved in
                    opposite directions relative to the vanilla baseline
  axis_divergences  the same test between neighbours along one axis (lambda)
  plot_*            the figures, one question each

Plot rules (from the project's dataviz conventions): one y-axis per panel -- a
second measure gets its own panel, never a twin axis; at most three hues, the
first three slots of a CVD-validated categorical palette (they clear the
all-pairs gates, so they are safe in scatters too); colour follows the entity
(an init, a loss, a path), never its rank; solid hairline grid; a legend
whenever two or more series share a panel.
"""
from __future__ import annotations

import json

import numpy as np

# First three slots of the reference categorical palette, light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# UNALIGNED on Synthetic -- aligning erases the affine bias Axis C injects
# (tests/test_depth_semantics.py). The official Chamfer on DTU.
PRIMARY_GEOMETRY = {"blender": "unaligned_mae", "dtu": "chamfer"}
GEOMETRY_LABEL = {"unaligned_mae": "depth MAE, unaligned",
                  "unaligned_absrel": "depth AbsRel, unaligned",
                  "chamfer": "Chamfer distance",
                  "floater_frac": "floater fraction"}

DEFAULTS = {"dataset": "blender", "split": "fps", "depth_model": "gt",
            "depth_align": "gt", "sigma_rel": 0.0, "noise_corr_px": 0.0,
            "scale": 1.0, "shift": 0.0, "scale_jitter": 0.0, "shift_jitter": 0.0,
            "depth_kind": "ssi", "depth_lambda": 0.0, "sh_degree": 0,
            "iterations": 7000}
BASELINE_KEYS = ["dataset", "scene", "n_views", "split", "iterations", "sh_degree"]

INITS = ["sfm", "random", "depth"]
KINDS = ["ssi", "pearson", "absolute"]
PATHS = ["init only", "loss only", "init + loss"]
ARMS = {"noise": "sigma_rel", "global affine": "scale",
        "per-image affine": "scale_jitter"}
MARKERS = {"clean": "o", "noise": "s", "global affine": "^",
           "per-image affine": "D"}

TRADE_UP = "trade: looks better, geometry worse"
TRADE_DOWN = "trade: geometry better, looks worse"


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------
def prior_path(init, lam):
    """How the depth prior entered the run: init, loss, both, or not at all."""
    d, loss = init == "depth", float(lam) > 0
    if d and loss:
        return "init + loss"
    if d:
        return "init only"
    return "loss only" if loss else "none"


def corruption_of(row):
    if row["depth_model"] != "gt":
        return f"real ({row['depth_align']}-aligned)"
    on = [name for name, flag in (
        ("noise", row["sigma_rel"] > 0),
        ("global affine", row["scale"] != 1.0 or row["shift"] != 0.0),
        ("per-image affine", row["scale_jitter"] > 0 or row["shift_jitter"] > 0)) if flag]
    return "clean" if not on else " + ".join(on)


def classify(d_psnr, d_geom_rel, tol_psnr=0.1, tol_geom=0.02):
    """Name the effect of a change: appearance by PSNR (dB), geometry by the
    relative change of the geometry error (positive = worse)."""
    if not (np.isfinite(d_psnr) and np.isfinite(d_geom_rel)):
        return "no baseline"
    app = 1 if d_psnr > tol_psnr else (-1 if d_psnr < -tol_psnr else 0)
    geo = 1 if d_geom_rel < -tol_geom else (-1 if d_geom_rel > tol_geom else 0)
    if app == 1 and geo == -1:
        return TRADE_UP
    if app == -1 and geo == 1:
        return TRADE_DOWN
    if app >= 0 and geo >= 0 and (app or geo):
        return "helps"
    if app <= 0 and geo <= 0 and (app or geo):
        return "hurts"
    return "neutral"


def load_results(src, geometry=None, tol_psnr=0.1, tol_geom=0.02):
    """results.csv (or a DataFrame) -> DataFrame with path/corruption/deltas."""
    import pandas as pd

    df = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    for col, val in DEFAULTS.items():
        df[col] = df[col].fillna(val) if col in df else val
    df["path"] = [prior_path(i, lam) for i, lam in zip(df["init"], df["depth_lambda"])]
    df["corruption"] = df.apply(corruption_of, axis=1)
    df["geometry"] = np.nan
    if geometry:
        df["geometry"] = df[geometry]
    else:
        for ds, col in PRIMARY_GEOMETRY.items():
            if col in df:
                sel = df["dataset"] == ds
                df.loc[sel, "geometry"] = df.loc[sel, col]
    return add_deltas(df, tol_psnr, tol_geom)


def add_deltas(df, tol_psnr=0.1, tol_geom=0.02):
    """Deltas vs vanilla 3DGS: random init, no depth, same scene/views/budget."""
    keys = [k for k in BASELINE_KEYS if k in df]
    base_rows = df[(df["init"] == "random") & (df["path"] == "none")]
    base = (base_rows.groupby(keys, dropna=False)[["psnr", "geometry"]].mean()
            .rename(columns=lambda c: f"{c}_base").reset_index())
    out = df.drop(columns=[c for c in ("psnr_base", "geometry_base") if c in df])
    out = out.merge(base, on=keys, how="left")
    out["d_psnr"] = out["psnr"] - out["psnr_base"]
    out["d_geometry"] = out["geometry"] - out["geometry_base"]
    out["d_geometry_rel"] = out["d_geometry"] / out["geometry_base"].abs()
    out["effect"] = [classify(p, g, tol_psnr, tol_geom)
                     for p, g in zip(out["d_psnr"], out["d_geometry_rel"])]
    return out


def divergences(df):
    """Runs that traded one metric family for the other."""
    cols = [c for c in ("stage", "dataset", "scene", "n_views", "init", "depth_kind",
                        "depth_lambda", "corruption", "prior_absrel", "psnr",
                        "geometry", "d_psnr", "d_geometry_rel", "effect") if c in df]
    out = df[df["effect"].str.startswith("trade")][cols]
    return out.sort_values("d_psnr", ascending=False, key=np.abs).reset_index(drop=True)


def axis_divergences(df, axis="depth_lambda", tol_psnr=0.1, tol_geom=0.02,
                     group=("dataset", "scene", "n_views", "init", "depth_kind",
                            "corruption")):
    """Opposite moves of appearance and geometry between neighbours on `axis`.

    For the lambda axis the lambda=0 runs (whose depth_kind is canonical) are
    shared by every loss kind, so each kind's curve starts from the same point.
    """
    import pandas as pd

    d = df.dropna(subset=["psnr", "geometry"])
    if axis == "depth_lambda":
        zero = d[d["depth_lambda"] == 0]
        d = pd.concat([zero.assign(depth_kind=k) for k in KINDS]
                      + [d[d["depth_lambda"] > 0]], ignore_index=True)
    group = [g for g in group if g != axis and g in d]
    rows = []
    for key, g in d.groupby(group, dropna=False):
        m = g.groupby(axis)[["psnr", "geometry"]].mean().sort_index()
        steps = m.index.tolist()
        for a, b in zip(steps[:-1], steps[1:]):
            dp = m.loc[b, "psnr"] - m.loc[a, "psnr"]
            dg = (m.loc[b, "geometry"] - m.loc[a, "geometry"]) / abs(m.loc[a, "geometry"])
            eff = classify(dp, dg, tol_psnr, tol_geom)
            if eff.startswith("trade"):
                rows.append({**dict(zip(group, key)), "axis": axis, "from": a, "to": b,
                             "d_psnr": dp, "d_geometry_rel": dg, "effect": eff})
    return pd.DataFrame(rows)


def iters_to_target(df):
    """Iterations until the probe PSNR first reaches vanilla's FINAL probe PSNR.

    NaN when a run never gets there. The probe curve is on 4 held-out views,
    so the target is the baseline's last probe value, not its full-eval PSNR.
    """
    curves = [json.loads(c) if isinstance(c, str) else None
              for c in df.get("psnr_curve", [None] * len(df))]
    finals = [c[-1][1] if c else np.nan for c in curves]
    d = df.assign(_curve=curves, _final=finals)
    keys = [k for k in BASELINE_KEYS if k in d]
    target = (d[(d["init"] == "random") & (d["path"] == "none")]
              .groupby(keys, dropna=False)["_final"].mean().rename("_target").reset_index())
    d = d.merge(target, on=keys, how="left")
    out = []
    for c, t in zip(d["_curve"], d["_target"]):
        hit = [s for s, p in (c or []) if np.isfinite(t) and p >= t]
        out.append(hit[0] if hit else np.nan)
    return np.array(out, dtype=float)


def summary(df, by, metrics=("psnr", "ssim", "lpips", "geometry")):
    """Mean +- std over scenes, one row per setting."""
    m = [c for c in metrics if c in df]
    g = df.groupby(list(by), dropna=False)[m]
    out = g.mean().round(4)
    out.columns = [f"{c}_mean" for c in out.columns]
    sd = g.std().round(4)
    sd.columns = [f"{c}_std" for c in sd.columns]
    return out.join(sd).join(g.size().rename("n"))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 9.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "legend.frameon": False, "axes.titlesize": 10.5,
        "axes.titlelocation": "left", "lines.solid_capstyle": "round",
    })
    return plt


def _style(ax, xlabel, ylabel, title=None):
    ax.grid(True, color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, color=INK)


def _geom_label(df):
    """Label from the datasets actually plotted: Blender runs also carry a
    'chamfer' column, but their primary geometry number is depth MAE."""
    cols = [PRIMARY_GEOMETRY[d] for d in sorted(set(df.get("dataset", [])))
            if d in PRIMARY_GEOMETRY]
    name = " / ".join(GEOMETRY_LABEL.get(c, c) for c in cols) or "geometry error"
    return f"{name} (lower is better)"


class NoData(str):
    """Returned instead of a figure when the runs it needs do not exist yet.

    Empty axes read as a broken plot; this names the stage that is missing.
    """


def _empty(fig, msg):
    import matplotlib.pyplot as plt
    plt.close(fig)          # otherwise the notebook still auto-displays it
    return NoData(msg)


# What must match for two runs to sit on the same line of a figure.
PROTOCOL_KEYS = ["split", "sh_degree", "iterations", "downscale"]


def _one_protocol(d):
    """Keep the runs of a single protocol -- the one with the most runs.

    stage0_repro shares the axes of stage1 (init, view count, no depth loss)
    but runs the published split, SH degree 3 and a 6k budget. Drawing its
    points on the stage1 lines would connect runs that differ in more than the
    axis being plotted.
    """
    keys = [k for k in PROTOCOL_KEYS if k in d]
    if not keys or d.empty:
        return d
    counts = d.groupby(keys, dropna=False).size()
    if len(counts) <= 1:
        return d
    top = counts.idxmax()
    top = top if isinstance(top, tuple) else (top,)
    mask = np.ones(len(d), dtype=bool)
    for k, v in zip(keys, top):
        mask &= (d[k] == v).to_numpy()
    return d[mask]


def _protocol_label(d):
    bits = []
    for k, fmt in (("split", "{} split"), ("n_views", None), ("sh_degree", "SH {}"),
                   ("iterations", "{} iters"), ("downscale", None)):
        if fmt and k in d and d[k].nunique() == 1:
            bits.append(fmt.format(d[k].iloc[0]))
    return ", ".join(bits)


def _line(ax, x, y, color, label):
    ax.plot(x, y, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
            label=label, solid_joinstyle="round")


def plot_init_by_views(df, dataset="blender"):
    """Axis A x B: each initialization, without any depth loss, vs view count."""
    plt = _plt()
    d = df[(df["dataset"] == dataset) & (df["depth_lambda"] == 0)
           & (df["corruption"] == "clean")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    if d.empty:
        return _empty(fig, "needs runs without a depth loss (stage1)")
    d = _one_protocol(d)
    for ax, metric, ylab in ((axes[0], "psnr", "PSNR, dB (higher is better)"),
                             (axes[1], "geometry", _geom_label(d))):
        for k, init in enumerate(INITS):
            s = d[d["init"] == init].dropna(subset=[metric])
            if s.empty:
                continue
            ax.scatter(s["n_views"], s[metric], s=14, color=SERIES[k], alpha=0.35, lw=0)
            m = s.groupby("n_views")[metric].mean()
            _line(ax, m.index, m.values, SERIES[k], init)
        ax.set_xticks(sorted(d["n_views"].unique()))
        _style(ax, "input views", ylab)
    axes[0].set_title("Appearance", color=INK)
    axes[1].set_title("Geometry", color=INK)
    axes[0].legend(title="initialization", loc="best")
    fig.suptitle("Initialization vs view count, no depth loss — mean over scenes, "
                 f"faint dots are single scenes ({_protocol_label(d)})",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    return fig


def plot_lambda_sweep(df, init="depth", n_views=3, dataset="blender"):
    """Where depth flips from help to harm: each loss kind vs lambda."""
    plt = _plt()
    d = df[(df["dataset"] == dataset) & (df["init"] == init)
           & (df["n_views"] == n_views) & (df["corruption"] == "clean")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    lam = d[d["depth_lambda"] > 0]
    if lam["depth_lambda"].nunique() < 2:
        return _empty(fig, f"needs at least two depth-loss weights for init={init}, "
                           f"{n_views} views (stage2_lambda)")
    zero = d[d["depth_lambda"] == 0]
    for ax, metric, ylab in ((axes[0], "psnr", "PSNR, dB (higher is better)"),
                             (axes[1], "geometry", _geom_label(d))):
        for k, kind in enumerate(KINDS):
            s = lam[lam["depth_kind"] == kind].dropna(subset=[metric])
            if s.empty:
                continue
            ax.scatter(s["depth_lambda"], s[metric], s=14, color=SERIES[k],
                       alpha=0.35, lw=0)
            m = s.groupby("depth_lambda")[metric].mean()
            _line(ax, m.index, m.values, SERIES[k], kind)
        if not zero[metric].dropna().empty:
            y0 = zero[metric].mean()
            ax.axhline(y0, color=MUTED, lw=1)
            ax.annotate("λ = 0", (1.0, y0), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", ha="right",
                        color=MUTED, fontsize=8.5)
        ax.set_xscale("log")
        _style(ax, "depth-loss weight λ", ylab)
    axes[0].set_title("Appearance", color=INK)
    axes[1].set_title("Geometry", color=INK)
    axes[0].legend(title="depth loss", loc="best")
    fig.suptitle(f"λ sweep, init={init}, {n_views} views (clean prior)",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    return fig


def plot_degradation(df, kind="absolute", n_views=3, dataset="blender"):
    """Axis C: outcome vs how wrong the prior is, one column per corruption arm.

    The x-axis is prior_absrel -- the prior's unaligned relative depth error on
    the input views -- so every arm, and the real model, share one axis. Series
    are the paths the prior takes into the run; the loss-bearing paths use
    `kind`. Stars mark the real monocular model.
    """
    plt = _plt()
    from matplotlib.lines import Line2D

    d = df[(df["dataset"] == dataset) & (df["n_views"] == n_views)]
    arms = [a for a in ARMS if (d["corruption"] == a).any()]
    fig, axes = plt.subplots(2, max(len(arms), 1), figsize=(4 * max(len(arms), 1), 6.6),
                             squeeze=False, sharey="row", constrained_layout=True)
    if not arms or "prior_absrel" not in d:
        return _empty(fig, "needs controlled-degradation runs (stage3_*)")

    real = d[d["depth_model"] != "gt"]
    for c, arm in enumerate(arms):
        for r, (metric, ylab) in enumerate((("psnr", "PSNR, dB (higher is better)"),
                                            ("geometry", _geom_label(d)))):
            ax = axes[r, c]
            for k, path in enumerate(PATHS):
                sel = d[(d["path"] == path) & (d["depth_model"] == "gt")]
                if path != "init only":
                    sel = sel[sel["depth_kind"] == kind]
                arm_rows = sel[sel["corruption"] == arm]
                if arm_rows.empty:
                    continue
                sel = sel[sel["corruption"].isin([arm, "clean"])
                          & sel["depth_lambda"].isin(arm_rows["depth_lambda"].unique())
                          & sel["init"].isin(arm_rows["init"].unique())]
                knob = ARMS[arm]
                m = (sel.groupby(knob)[["prior_absrel", metric]].mean()
                     .dropna().sort_values("prior_absrel"))
                if not m.empty:
                    _line(ax, m["prior_absrel"], m[metric], SERIES[k], path)
                rp = real[real["path"] == path]
                if path != "init only":
                    rp = rp[rp["depth_kind"] == kind]
                for _, g in rp.groupby("depth_align"):
                    ax.scatter(g["prior_absrel"].mean(), g[metric].mean(), marker="*",
                               s=170, color=SERIES[k], edgecolor=SURFACE, lw=1.2, zorder=5)
            _style(ax, "prior error: AbsRel vs GT, unaligned" if r else "",
                   ylab if c == 0 else "", arm if r == 0 else None)

    handles = [Line2D([], [], color=SERIES[k], lw=2, marker="o", ms=6, mec=SURFACE,
                      label=p) for k, p in enumerate(PATHS)]
    if not real.empty:
        handles.append(Line2D([], [], ls="", marker="*", ms=11, color=MUTED,
                              label="Depth Anything V2"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(f"Controlled degradation, {n_views} views, loss = {kind} "
                 "(x = how wrong the prior is)", x=0.01, ha="left", color=INK2,
                 fontsize=9)
    return fig


def plot_tradeoff_map(df, dataset="blender"):
    """Every run with a depth prior, as a change vs vanilla 3DGS.

    Right = looks better, up = geometry worse: the shaded upper-right quadrant
    is where a prior flatters the renders while the geometry degrades.
    """
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    d = df[(df["dataset"] == dataset) & (df["path"] != "none")].dropna(
        subset=["d_psnr", "d_geometry_rel"])
    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    if d.empty:
        return _empty(fig, "needs a vanilla-3DGS baseline to compare against "
                           "(random init, no depth loss: stage1)")

    # A single blown-up run (a large lambda on a scale-invariant loss can be
    # several hundred percent worse) would squash every other point into a
    # band. Scale to the bulk and park the rest on the frame as triangles.
    x_all = d["d_psnr"].to_numpy(dtype=float)
    y_all = 100 * d["d_geometry_rel"].to_numpy(dtype=float)
    lo, hi = (np.nanpercentile(y_all, [2, 98]) if len(y_all) > 4
              else (np.nanmin(y_all), np.nanmax(y_all)))
    pad = 0.1 * max(hi - lo, 1.0)
    lo, hi = min(lo - pad, -5.0), max(hi + pad, 5.0)
    n_out = int((y_all > hi).sum() + (y_all < lo).sum())

    def draw(sel, marker, color, size, z):
        if not len(sel):
            return
        x = sel["d_psnr"].to_numpy(dtype=float)
        y = 100 * sel["d_geometry_rel"].to_numpy(dtype=float)
        keep = (y >= lo) & (y <= hi)
        ax.scatter(x[keep], y[keep], marker=marker, s=size, color=color,
                   edgecolor=SURFACE, lw=1, zorder=z)
        for outside, edge, mk in ((y > hi, hi, "^"), (y < lo, lo, "v")):
            if outside.any():
                ax.scatter(x[outside], np.full(int(outside.sum()), edge), marker=mk,
                           s=size, color=color, edgecolor=SURFACE, lw=1, alpha=0.75,
                           zorder=z)

    for k, path in enumerate(PATHS):
        for corr, mk in MARKERS.items():
            draw(d[(d["path"] == path) & (d["corruption"] == corr)], mk, SERIES[k], 42, 3)
        draw(d[(d["path"] == path) & d["corruption"].str.startswith("real")],
             "*", SERIES[k], 150, 4)

    xlo, xhi = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    xpad = 0.1 * max(xhi - xlo, 1.0)
    ax.set_xlim(min(xlo - xpad, -0.5), max(xhi + xpad, 0.5))
    ax.set_ylim(lo, hi)
    x1 = ax.get_xlim()[1]
    ax.add_patch(Rectangle((0, 0), x1, hi, color=SERIES[1], alpha=0.07, lw=0, zorder=0))
    ax.text(x1, hi, "looks better,\ngeometry worse", ha="right", va="top",
            color=INK2, fontsize=8.5)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    _style(ax, "Δ PSNR vs vanilla 3DGS, dB (right = looks better)",
           "Δ geometry error vs vanilla, % (up = worse)")

    handles = [Line2D([], [], ls="", marker="o", ms=7, color=SERIES[k], label=p)
               for k, p in enumerate(PATHS)]
    handles += [Line2D([], [], ls="", marker=mk, ms=7, color=MUTED, label=c)
                for c, mk in MARKERS.items() if (d["corruption"] == c).any()]
    if d["corruption"].str.startswith("real").any():
        handles.append(Line2D([], [], ls="", marker="*", ms=10, color=MUTED,
                              label="real model"))
    fig.legend(handles=handles, loc="outside lower center",
               ncol=min(4, len(handles)))
    note = (f"; {n_out} beyond the axis, drawn at the frame" if n_out else "")
    fig.suptitle(f"Every run with a depth prior, vs vanilla 3DGS{note}",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    return fig


def save(fig, path_stem):
    """PNG for the notebook and slides, PDF for the report. False if skipped.

    A skipped figure also removes its own earlier output, so the figures
    folder never holds a stale placeholder from a session with fewer runs.
    """
    from pathlib import Path

    if isinstance(fig, NoData):
        for ext in ("png", "pdf"):
            Path(f"{path_stem}.{ext}").unlink(missing_ok=True)
        print(f"skipped {Path(str(path_stem)).name}: {fig}")
        return False
    fig.savefig(f"{path_stem}.png", dpi=200)
    fig.savefig(f"{path_stem}.pdf")
    return True
