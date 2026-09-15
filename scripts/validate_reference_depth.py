"""Measure the NeRF-Synthetic reference depth against exact depth from Blender.

    python scripts/validate_reference_depth.py --exact exact_depth \
        --scene lego chair ship --out results/reference_validation.json

--exact holds <scene>/<split>/r_<i>.npz from scripts/render_blender_depth.py;
the reference is <reference-root>/<scene>/depth_<split>/r_<i>.npy from
scripts/make_gt_depth.py, and the release RGBA images come from $DATA_ROOT.

The study never saw exact depth: its "clean" prior is a 400-view
reconstruction, and every corruption level is reported as the prior's AbsRel
against it. The number that decides whether that was sound is the
reference's own AbsRel against exact depth, computed the same way the study
computes prior error (2x area-downsampled with downscale_depth, unaligned,
mean over views) -- if it is far below the smallest corruption swept, the
degradation curves measured prior error and not reference error.

Two checks come first, because they decide whether the .blend geometry is
the geometry the release images show at all:
  silhouette_iou   Blender's coverage vs the release alpha (> 0.5)
  png_spearman     exact depth vs the release's 8-bit depth PNGs (test split)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.data import downscale_depth, load_blender_scene, resolve_data_root  # noqa: E402
from src.eval_geometry import depth_metrics                               # noqa: E402

STUDY_DOWNSCALE = 2


def view_agreement(reference, exact, release_mask, exact_alpha,
                   downscale=STUDY_DOWNSCALE):
    """Reference-vs-exact statistics for one view (0 = no depth in both maps).

    Full-resolution relative errors are pooled over pixels valid in both maps;
    absrel/mae repeat the study's prior-error computation at its resolution.
    """
    ref = np.asarray(reference, dtype=np.float64)
    ex = np.asarray(exact, dtype=np.float64)
    mask = np.asarray(release_mask, dtype=bool)
    silhouette = np.asarray(exact_alpha) > 0.5
    union = (silhouette | mask).sum()

    both = (ref > 0) & (ex > 0)
    rel = np.abs(ref[both] - ex[both]) / ex[both]
    small_r = downscale_depth(ref, downscale)
    small_e = downscale_depth(ex, downscale)
    scored = (small_r > 0) & (small_e > 0)
    m = depth_metrics(small_r, small_e, mask=scored)
    return {
        "silhouette_iou": float((silhouette & mask).sum() / union) if union else float("nan"),
        "coverage": float(both.sum() / max(((ex > 0) & mask).sum(), 1)),
        "rel": rel,
        "absrel": m["absrel"], "mae": m["mae"],
    }


def summarize(rows):
    rel = np.concatenate([r["rel"] for r in rows])
    return {
        "n_views": len(rows),
        "absrel": float(np.mean([r["absrel"] for r in rows])),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "median_rel": float(np.median(rel)),
        "p90_rel": float(np.percentile(rel, 90)),
        "within_1pct": float((rel < 0.01).mean()),
        "silhouette_iou": float(np.mean([r["silhouette_iou"] for r in rows])),
        "coverage": float(np.mean([r["coverage"] for r in rows])),
    }


def plot_reference_error(exact_root, reference_root, report, view=0, limit=0.05):
    """Signed relative error of the reference on one evaluation view per scene."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.ticker import PercentFormatter

    from src import analysis as A

    plt = A._plt()
    scenes = list(report["scenes"])
    fig, axes = plt.subplots(1, len(scenes), figsize=(2.5 * len(scenes) + 1.0, 2.9),
                             constrained_layout=True, squeeze=False)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(A.SURFACE)
    for ax, scene in zip(axes[0], scenes):
        ex = np.load(exact_root / scene / "test" / f"r_{view}.npz")["depth"].astype(np.float64)
        ref = np.load(reference_root / scene / "depth_test" / f"r_{view}.npy").astype(np.float64)
        both = (ref > 0) & (ex > 0)
        err = np.full(ex.shape, np.nan)
        err[both] = (ref[both] - ex[both]) / ex[both]
        # A square crop around the object, so every panel's title lines up.
        rows, cols = np.nonzero(both)
        half = (max(np.ptp(rows), np.ptp(cols)) + 1) // 2 + 12
        r0 = (rows.min() + rows.max()) // 2 - half
        c0 = (cols.min() + cols.max()) // 2 - half
        crop = np.full((2 * half, 2 * half), np.nan)
        src_r = slice(max(r0, 0), min(r0 + 2 * half, err.shape[0]))
        src_c = slice(max(c0, 0), min(c0 + 2 * half, err.shape[1]))
        crop[src_r.start - r0: src_r.stop - r0, src_c.start - c0: src_c.stop - c0] = err[src_r, src_c]
        im = ax.imshow(crop, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
        absrel = report["scenes"][scene]["test"]["absrel"]
        ax.set_title(f"{scene}  (AbsRel {absrel:.2%})", loc="center")
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85,
                      format=PercentFormatter(1.0, decimals=0))
    cb.set_label("(reference − exact) / exact", color=A.INK2)
    cb.outline.set_visible(False)
    return fig


def validate_scene(scene, exact_root, reference_root):
    from make_gt_depth import png_rank_correlation

    out = {}
    scene_dir = resolve_data_root() / "nerf_synthetic" / scene
    for split_dir in sorted((exact_root / scene).iterdir()):
        split = split_dir.name
        ids = sorted(int(p.stem[2:]) for p in split_dir.glob("r_*.npz"))
        if not ids:
            continue
        views = load_blender_scene(scene=scene, split=split, load_depth=False,
                                   indices=ids).views
        rows, exact_views, ref_views = [], [], []
        for i, v in zip(ids, views):
            npz = np.load(split_dir / f"r_{i}.npz")
            ref = np.load(reference_root / scene / f"depth_{split}" / f"r_{i}.npy")
            rows.append(view_agreement(ref, npz["depth"], v.mask, npz["alpha"]))
            exact_views.append(SimpleNamespace(name=v.name, depth=npz["depth"]))
            ref_views.append(SimpleNamespace(name=v.name, depth=ref))
        res = summarize(rows)
        if split == "test":
            res["png_spearman_exact"] = png_rank_correlation(scene_dir, exact_views)["png_spearman"]
            res["png_spearman_reference"] = png_rank_correlation(scene_dir, ref_views)["png_spearman"]
        out[split] = res
        print(f"[{scene}/{split}] {len(ids)} views  AbsRel {res['absrel']:.3%}  "
              f"median {res['median_rel']:.3%}  within 1% {res['within_1pct']:.1%}  "
              f"silhouette IoU {res['silhouette_iou']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exact", required=True, help="output root of render_blender_depth.py")
    ap.add_argument("--scene", nargs="+", required=True)
    ap.add_argument("--reference-root", default=None,
                    help="directory holding <scene>/depth_<split>/ "
                         "(default: $DATA_ROOT/nerf_synthetic)")
    ap.add_argument("--smallest-corruption", type=float, default=0.039,
                    help="smallest prior AbsRel the study swept, for the ratio")
    ap.add_argument("--out", default="results/reference_validation.json")
    ap.add_argument("--figure", default=None,
                    help="path stem for a PNG+PDF of the signed error on one eval view")
    args = ap.parse_args()

    ref_root = Path(args.reference_root or resolve_data_root() / "nerf_synthetic")
    report = {"smallest_corruption_absrel": args.smallest_corruption, "scenes": {}}
    for scene in args.scene:
        res = validate_scene(scene, Path(args.exact), ref_root)
        worst = max(split["absrel"] for split in res.values())
        res["margin"] = args.smallest_corruption / worst
        report["scenes"][scene] = res
        print(f"[{scene}] reference AbsRel up to {worst:.2%}: {res['margin']:.1f}x below "
              f"the smallest corruption ({args.smallest_corruption:.1%})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    if args.figure:
        from src.analysis import save
        save(plot_reference_error(Path(args.exact), ref_root, report), args.figure)


if __name__ == "__main__":
    main()
