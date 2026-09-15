"""Draw the report figures that src/analysis.py does not.

    python scripts/make_report_figures.py --results results/results.csv \
        --out report/figures

report_init_views     PSNR and reference-depth MAE against view count
report_lambda_grid    clean-prior depth-loss sweep at three and nine views
report_corruption     matched three-view corruption comparison (Lego, Chair)
report_dtu_scans      DTU Chamfer per scan, reference init with and without SSI

Every figure is built from results.csv alone, so it can be redrawn on a laptop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import pandas as pd                        # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

COLORS = ("#2f5d8a", "#c0582d", "#4a8a5c")
MARKERS = ("o", "s", "^")
PAIR = ("lego", "chair")                   # the scenes of the corruption stages


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def _line(ax, x, y, k, label=None):
    ax.plot(list(x), list(y), color=COLORS[k], marker=MARKERS[k], lw=1.6, ms=6,
            label=label)


def init_views(df):
    means = (df[df.stage == "stage1"]
             .groupby(["init", "n_views"])[["psnr", "unaligned_mae"]].mean())
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for k, (init, label) in enumerate((("depth", "reference depth"), ("sfm", "SfM"),
                                       ("random", "random"))):
        m = means.loc[init]
        _line(a, m.index, m.psnr, k, label)
        _line(b, m.index, m.unaligned_mae, k)
    a.set_ylabel("PSNR (dB)")
    b.set_ylabel("Reference-depth MAE")
    for ax in (a, b):
        ax.set_xticks([3, 5, 9])
        ax.set_xlabel("Training views")
    a.legend()
    fig.tight_layout()
    return fig


def lambda_grid(df):
    means = (df[df.stage == "stage2_lambda"]
             .groupby(["n_views", "depth_kind", "depth_lambda"])[["psnr", "unaligned_mae"]]
             .mean())
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8))
    for r, views in enumerate((3, 9)):
        for k, (kind, label) in enumerate((("ssi", "SSI"), ("pearson", "Pearson"),
                                           ("absolute", "Absolute"))):
            m = means.loc[(views, kind)]
            _line(axes[r, 0], m.index, m.psnr, k, label)
            _line(axes[r, 1], m.index, m.unaligned_mae, k)
        for c, ylabel in enumerate(("PSNR (dB)", "Reference-depth MAE")):
            ax = axes[r, c]
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks([0.03, 0.1, 0.3, 1.0], ["0.03", "0.1", "0.3", "1"])
            ax.set_title(f"{views} views")
            ax.set_xlabel(r"Depth-loss weight $\lambda$")
            ax.set_ylabel(ylabel)
    axes[0, 0].legend()
    fig.tight_layout()
    return fig


def corruption(df):
    """Depth init + SSI at lambda 0.1, against random init without depth."""
    base = df[(df.stage == "stage1") & (df.n_views == 3) & (df.init == "random")
              & df.scene.isin(PAIR)][["psnr", "unaligned_mae"]].mean()
    arm = df[df.stage.str.startswith("stage3") & (df.init == "depth")
             & (df.depth_kind == "ssi") & (df.depth_lambda == 0.1)]
    series = (("noise", "stage3_noise", "sigma_rel", None),
              ("global scale", "stage3_affine", "scale", (0.8, 1.5)),
              ("per-image affine", "stage3_perimage", "scale_jitter", (0.05, 0.2)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for k, (label, stage, knob, keep) in enumerate(series):
        d = arm[arm.stage == stage]
        if keep is not None:
            d = d[d[knob].isin(keep)]
        m = d.groupby(knob)[["prior_absrel", "psnr", "unaligned_mae"]].mean()
        _line(a, 100 * m.prior_absrel, m.psnr, k, label)
        _line(b, 100 * m.prior_absrel, m.unaligned_mae, k)
    a.axhline(base.psnr, color="black", ls="--", lw=1, label="matched random baseline")
    b.axhline(base.unaligned_mae, color="black", ls="--", lw=1)
    a.set_ylabel("PSNR (dB)")
    b.set_ylabel("Reference-depth MAE")
    for ax in (a, b):
        ax.set_xlabel("Prior AbsRel to reference (%)")
    a.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    return fig


def dtu_scans(df):
    d = df[df.stage == "stage5_dtu_oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, views in zip(axes, (3, 9)):
        for k, scan in enumerate(("scan1", "scan6")):
            s = d[(d.n_views == views) & (d.scene == scan)].sort_values("depth_lambda")
            _line(ax, (0, 1), s.chamfer, k, f"Scan {scan[4:]}")
        ax.set_xticks([0, 1], ["Init only", "Init + SSI"])
        ax.set_xlim(-0.15, 1.15)
        ax.set_title(f"{views} views")
    axes[0].set_ylabel("DTU Chamfer (mm)")
    axes[0].legend()
    fig.tight_layout()
    return fig


FIGURES = {"report_init_views": init_views, "report_lambda_grid": lambda_grid,
           "report_corruption": corruption, "report_dtu_scans": dtu_scans}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()
    for name, draw in FIGURES.items():
        fig = draw(df)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}")
        plt.close(fig)
        print("wrote", out / name)


if __name__ == "__main__":
    main()
