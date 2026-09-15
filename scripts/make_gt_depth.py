"""Build the NeRF-Synthetic reference depth that Axis C perturbs.

    python scripts/make_gt_depth.py --scene lego chair ship \
        --persist "/content/drive/MyDrive/datasets/nerf_synthetic_depth"

WHY A RECONSTRUCTION AND NOT THE .blend FILES. NeRF-Synthetic ships no metric
depth, neither the NeRF Google Drive archive (re-uploaded 2024) nor the
nerfbaselines mirror contains a .blend file, and the test split's
r_*_depth_*.png are 8-bit, clipped, and not metric. The modified scenes do
circulate through the NeRF issue tracker (bmild/nerf #59, #198), but we found
them only after the sweep, so they validate this reference rather than replace
it: scripts/render_blender_depth.py renders exact depth from them and
scripts/validate_reference_depth.py measures the reference against it.

The reference comes from the densest reconstruction the data supports: 3DGS
trained on ALL 400 views of
the scene (train + val + test) at full 800x800 resolution, additionally
supervised by the exact alpha masks the RGBA images carry, rendered as
expected depth at every train and test pose and masked by the true alpha.

What keeps it sound for this study:

  * Corruptions are applied relative to the reference, so the
    question "how does prior error propagate into geometry" is unchanged --
    only the zero point is a dense oracle instead of the mesh.
  * <scene>/depth_reference.json records multi-view reprojection consistency
    between its depth maps and rank correlation against the 8-bit depth PNGs.
    Those check SELF-consistency, not accuracy: on ship they reported 0.17%
    while exact depth shows 2.3% AbsRel, because a water surface reconstructed
    in front of the true one is offset consistently in every view. The
    accuracy measurement is scripts/validate_reference_depth.py.

Output: <DATA_ROOT>/nerf_synthetic/<scene>/depth_{train,test}/r_<i>.npy
(float32, full resolution, depth along the camera's forward axis, 0 = background).
With --persist the depth folders are also zipped to that directory, so later
Colab sessions restore them in seconds instead of re-training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.data import load_blender_scene, resolve_data_root          # noqa: E402
from src.eval_geometry import reprojection_consistency               # noqa: E402
from src.gaussians import build_initial_gaussians                    # noqa: E402
from src.metrics import psnr                                         # noqa: E402
from src.render import render                                        # noqa: E402

WRITE_SPLITS = ("train", "test")          # what the study reads
FIT_SPLITS = ("train", "val", "test")     # what the reference is trained on


def reference_config(scene, iterations, alpha_weight, sh_degree):
    return {
        "dataset": "blender", "scene": scene, "background": "white",
        "init": "random", "init_opts": {"random_points": 100_000, "random_extent": 1.3},
        "loss": {"rgb_weight": 1.0, "depth_lambda": 0.0, "alpha_weight": alpha_weight},
        "train": {"iterations": iterations, "seed": 0, "sh_degree": sh_degree,
                  "densify_until": 0.5, "reset_every": 3000,
                  "probe_every": 10 ** 9},
    }


def png_rank_correlation(scene_dir, views, max_pixels=200_000, seed=0):
    """|Spearman rho| between reference depth and the release's 8-bit depth PNGs.

    The PNG encoding is an unknown monotone map of depth (and clipped), so
    only the ranking is comparable. A reference that orders pixels by depth
    the way Blender did scores close to 1.
    """
    import imageio.v2 as imageio
    from scipy.stats import spearmanr

    ref_vals, png_vals = [], []
    for v in views:
        cands = sorted((scene_dir / "test").glob(f"{v.name}_depth_*.png"))
        if not cands or v.depth is None:
            continue
        png = np.asarray(imageio.imread(cands[0]))
        val = png[..., 0] if png.ndim == 3 else png
        ok = (v.depth > 0) & (val > 0) & (val < 255)
        if png.ndim == 3 and png.shape[-1] == 4:
            ok &= png[..., 3] > 0
        ref_vals.append(v.depth[ok])
        png_vals.append(val[ok].astype(np.float64))
    if not ref_vals:
        return {"png_spearman": None}
    r, p = np.concatenate(ref_vals), np.concatenate(png_vals)
    if len(r) > max_pixels:
        idx = np.random.default_rng(seed).choice(len(r), max_pixels, replace=False)
        r, p = r[idx], p[idx]
    return {"png_spearman": float(abs(spearmanr(r, p).correlation)),
            "png_pixels": int(len(r))}


def multiview_consistency(test_views, train_views):
    """Each test view against its nearest training camera."""
    centres = np.stack([v.c2w[:3, 3] for v in train_views])
    meds, within = [], []
    for v in test_views:
        nb = train_views[int(np.argmin(np.linalg.norm(centres - v.c2w[:3, 3], axis=1)))]
        r = reprojection_consistency(v.depth, v, nb.depth, nb)
        if r["n"]:
            meds.append(r["median_rel"])
            within.append(r["within_1pct"])
    return {"mv_median_rel_err": float(np.median(meds)) if meds else None,
            "mv_within_1pct": float(np.mean(within)) if within else None,
            "mv_pairs": len(meds)}


def build(scene, args, device):
    import train as train_mod

    root = resolve_data_root()
    scene_dir = root / "nerf_synthetic" / scene
    t0 = time.time()
    splits = {s: load_blender_scene(scene=scene, split=s, load_depth=False,
                                    background=1.0).views for s in FIT_SPLITS}
    views = [v for s in FIT_SPLITS for v in splits[s]]
    print(f"[{scene}] fitting the reference on {len(views)} views "
          f"({args.iterations} iterations)", flush=True)

    cfg = reference_config(scene, args.iterations, args.alpha_weight, args.sh_degree)
    g, _ = build_initial_gaussians(cfg, views, device=device)
    stats = train_mod.train(cfg, g, views, device, scene_dir)

    report = {"scene": scene, "iterations": args.iterations,
              "n_fit_views": len(views), "n_gaussians": len(g),
              "train_seconds": stats["train_seconds"]}
    with torch.no_grad():
        for split in WRITE_SPLITS:
            ddir = scene_dir / f"depth_{split}"
            ddir.mkdir(exist_ok=True)
            fits, cover = [], []
            for v in splits[split]:
                rgb, depth, alpha, _ = render(g, [v], device, background=1.0)
                d = depth[0].cpu().numpy()
                keep = (alpha[0, ..., 0].cpu().numpy() > 0.5) & np.isfinite(d) & (d > 0)
                if v.mask is not None:
                    cover.append(float((keep & v.mask).sum() / max(v.mask.sum(), 1)))
                    keep &= v.mask
                v.depth = np.where(keep, d, 0.0).astype(np.float32)
                np.save(ddir / f"{v.name}.npy", v.depth)
                fits.append(psnr(rgb[0].clamp(0, 1),
                                 torch.as_tensor(v.image, device=rgb.device)))
            report[f"fit_psnr_{split}"] = float(np.mean(fits))
            report[f"mask_coverage_{split}"] = float(np.mean(cover)) if cover else None

    probe = splits["test"][::8]
    report.update(multiview_consistency(probe, splits["train"]))
    report.update(png_rank_correlation(scene_dir, probe))
    report["wall_seconds"] = time.time() - t0
    (scene_dir / "depth_reference.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    if args.persist:
        dest = Path(args.persist)
        dest.mkdir(parents=True, exist_ok=True)
        zpath = dest / f"{scene}_depth.zip"
        tmp = zpath.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for split in WRITE_SPLITS:
                for f in sorted((scene_dir / f"depth_{split}").glob("*.npy")):
                    z.write(f, f"{scene}/depth_{split}/{f.name}")
            z.write(scene_dir / "depth_reference.json", f"{scene}/depth_reference.json")
        tmp.replace(zpath)
        print(f"  persisted -> {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")
    del g
    torch.cuda.empty_cache()


def restore(scene, persist):
    """Unzip a persisted reference into DATA_ROOT. True if it was there."""
    zpath = Path(persist) / f"{scene}_depth.zip"
    if not zpath.exists():
        return False
    with zipfile.ZipFile(zpath) as z:
        z.extractall(resolve_data_root() / "nerf_synthetic")
    print(f"[{scene}] restored reference depth from {zpath}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", nargs="+", required=True)
    ap.add_argument("--iterations", type=int, default=30_000)
    ap.add_argument("--alpha-weight", type=float, default=0.2,
                    help="weight of the L1 loss between rendered alpha and the true mask")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--persist", default=None,
                    help="directory (on Drive) to zip the depth maps into / restore from")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if depth maps or a persisted zip exist")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU: the reference reconstruction needs gsplat.")
    device = torch.device("cuda")
    for scene in args.scene:
        scene_dir = resolve_data_root() / "nerf_synthetic" / scene
        have = all(any((scene_dir / f"depth_{s}").glob("*.npy")) for s in WRITE_SPLITS)
        if not args.force and have:
            print(f"[{scene}] depth already present; --force to rebuild")
            continue
        if not args.force and args.persist and restore(scene, args.persist):
            continue
        build(scene, args, device)


if __name__ == "__main__":
    main()
