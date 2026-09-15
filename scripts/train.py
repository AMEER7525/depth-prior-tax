"""Train one depth-prior 3DGS run from a config, and write metrics.json.

The contract with scripts/run_sweep.py: read --config, write metrics.json into
--out. run_sweep treats metrics.json as the resume key, so it is written only
on success, and atomically. On failure the reason goes to error.txt, which
scripts/evaluate.py collects: a cell that cannot run (SfM triangulating nothing
from 3 views) is a result too.

Every run reports BOTH metric families. A run with appearance numbers only
cannot contribute to the appearance-vs-geometry map, which is the contribution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.corruptions import degrade_view                            # noqa: E402
from src.data import (DTU_TEST_VIEWS, blender_poses,                # noqa: E402
                      blender_sparse_split, dtu_sparse_split, evenly_spaced,
                      load_blender_scene, load_dtu_eval_assets, load_dtu_scene,
                      parse_scan_id, resolve_data_root)
from src.depth_loss import depth_loss                               # noqa: E402
from src.eval_geometry import (chamfer_components, depth_metrics,   # noqa: E402
                               dtu_chamfer, floater_stats, fuse_depth_maps,
                               subsample)
from src.gaussians import build_initial_gaussians, reset_opacity    # noqa: E402
from src.metrics import LPIPS, appearance_metrics, psnr, ssim_tensor  # noqa: E402
from src.render import PACKED, render, to_image                     # noqa: E402

# 3DGS defaults. The means lr is scaled by scene extent and decays
# exponentially to 1% over training (1.6e-4 -> 1.6e-6, as in vanilla 3DGS);
# the rest are constant.
LR = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
      "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
MEANS_LR_FINAL = 0.01
SSIM_WEIGHT = 0.2       # L_rgb = (1-w)*L1 + w*(1-SSIM), as in vanilla 3DGS
SH_UP_EVERY = 1000      # 3DGS raises the active SH degree every 1000 steps
PROBE_VIEWS = 4         # held-out views rendered for the convergence curve
SUPPORTED_MODELS = ("gt", "depth_anything_v2")
SFM_ALIGN_PREFERRED = 5   # below this, widen from view-observed to all in-frame points


def background_of(cfg):
    bg = cfg.get("background", "black")
    if isinstance(bg, str):
        if bg not in ("white", "black"):
            raise ValueError(f"background must be white | black | a number, got {bg!r}")
        return 1.0 if bg == "white" else 0.0
    return float(bg)


# --------------------------------------------------------------------------
def load_scene_and_split(cfg):
    """Return (train_views, eval_views, info)."""
    ds = cfg.get("dataset", "blender")
    n = int(cfg.get("n_views", 3))
    f = int(cfg.get("downscale", 1))
    if ds == "blender":
        scene, bg = cfg["scene"], background_of(cfg)
        ids = blender_sparse_split(n, blender_poses(scene=scene, split="train"),
                                   protocol=cfg.get("split", "fps"))
        n_test = len(blender_poses(scene=scene, split="test"))
        eval_ids = evenly_spaced(cfg.get("eval_views", 25), n_test)
        tr = load_blender_scene(scene=scene, split="train", downscale=f,
                                background=bg, indices=ids)
        te = load_blender_scene(scene=scene, split="test", downscale=f,
                                background=bg, indices=eval_ids)
        return tr.views, te.views, {"dataset": ds, "train_ids": ids,
                                    "eval_ids": eval_ids}
    if ds == "dtu":
        scan = parse_scan_id(cfg["scene"])
        sc = load_dtu_scene(scan_id=scan, downscale=f)
        ids = dtu_sparse_split(n)
        eval_ids = [i for i in DTU_TEST_VIEWS if i < len(sc)]
        return ([sc.views[i] for i in ids], [sc.views[i] for i in eval_ids],
                {"dataset": ds, "train_ids": ids, "eval_ids": eval_ids,
                 "scale_mat": sc.meta["scale_mat"],
                 "dtu_eval": load_dtu_eval_assets(scan_id=scan)})
    raise ValueError(f"unknown dataset {ds!r}")


def _depth_used(cfg):
    return (cfg.get("init") == "depth"
            or (cfg.get("loss") or {}).get("depth_lambda", 0.0) > 0)


def triangulate_if_needed(cfg, views, info=None):
    """SfM points, when init=sfm or a real depth model is aligned against them.

    On NeRF-Synthetic, features are detected on the native 800 px images, as a
    COLMAP run on the originals would be; the points are in world units, so
    they serve the downscaled run unchanged. At 3 views this is the difference
    between SfM initializing and SfM failing on most scenes.
    """
    dcfg = cfg.get("depth") or {}
    need = cfg.get("init") == "sfm" or (
        _depth_used(cfg) and dcfg.get("model", "gt") != "gt"
        and dcfg.get("align", "gt") == "sfm")
    if not need:
        return None
    from src.sfm import triangulate_views
    opts = dict(cfg.get("sfm") or {})
    full_res = opts.pop("full_res", True)
    src = views
    if (full_res and cfg.get("dataset", "blender") == "blender"
            and int(cfg.get("downscale", 1)) > 1 and info is not None):
        src = load_blender_scene(scene=cfg["scene"], split="train", load_depth=False,
                                 background=background_of(cfg),
                                 indices=info["train_ids"]).views
    pts, cols, stats = triangulate_views(src, **opts)
    print(f"  sfm: {stats['n_points']} points from {len(views)} views "
          f"(keypoints {stats['n_keypoints']})")
    return pts, cols, stats


def prior_error_stats(priors, cleans, masks):
    """How wrong the prior is, against clean GT, on the input views.

    This is the common x-axis of every degradation curve: a synthetic
    corruption and a real model are placed on the same axis by what they do to
    depth, not by the knob that produced it. 'aligned' removes the best
    per-image affine first, isolating the non-affine part of the error.
    """
    rows = []
    for p, g, m in zip(priors, cleans, masks):
        if p is None or g is None:
            continue
        g = np.asarray(g, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        ref = np.isfinite(g) & (g > 0)
        if m is not None:
            ref &= m
        scored = ref & np.isfinite(p) & (p > 0)
        if scored.sum() < 2:
            continue
        u = depth_metrics(p, g, mask=scored)
        a = depth_metrics(p, g, mask=scored, align=True)
        rows.append({"prior_absrel": u["absrel"], "prior_mae": u["mae"],
                     "prior_rmse": u["rmse"], "prior_aligned_absrel": a["absrel"],
                     "prior_coverage": float(scored.sum() / max(ref.sum(), 1))})
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def attach_depth(cfg, views, device, sfm=None, cache_dir=None):
    """Put the (possibly corrupted) prior depth on each training view.

    Returns a dict describing what was done, which goes into metrics.json --
    a degradation curve is unreadable without knowing what was actually applied.
    """
    dcfg = cfg.get("depth") or {}
    lcfg = cfg.get("loss") or {}
    model_name = dcfg.get("model", "gt")
    uses_init = cfg.get("init") == "depth"
    uses_loss = lcfg.get("depth_lambda", 0.0) > 0
    needs_metric = uses_init or (uses_loss and lcfg.get("depth_kind", "ssi") == "absolute")
    knobs = {k: dcfg.get(k, d) for k, d in (
        ("sigma_rel", 0.0), ("noise_corr_px", 0.0), ("scale", 1.0), ("shift", 0.0),
        ("scale_jitter", 0.0), ("shift_jitter", 0.0))}
    info = {"model": model_name, **knobs}

    # The prior reaches a run through exactly two paths: initialization and the
    # depth loss. With neither active there is nothing to prepare, and demanding
    # depth would block the runs that need none -- the same reasoning that
    # prunes cells in src/sweep.py.
    if not uses_init and not uses_loss:
        return {**info, "used": False}
    info["used"] = True
    if model_name not in SUPPORTED_MODELS:
        raise SystemExit(f"depth.model={model_name!r}; supported: {SUPPORTED_MODELS}")
    clean = [None if v.depth is None else np.asarray(v.depth, np.float32).copy()
             for v in views]

    if model_name == "gt":
        if any(c is None for c in clean):
            raise SystemExit(
                "depth.model is 'gt' but the views carry no depth.\n"
                "NeRF-Synthetic ships no metric depth. Build the reference depth "
                "once per scene:\n"
                "  python scripts/make_gt_depth.py --scene <scene>\n"
                "which writes $DATA_ROOT/nerf_synthetic/<scene>/depth_train/r_<i>.npy "
                "(and depth_test/).\n"
                "Runs with init=random|sfm and depth_lambda=0 need no depth and "
                "work without it.")
        seed = (cfg.get("train") or {}).get("seed", 0)
        g_noise = torch.Generator().manual_seed(seed)
        g_jitter = torch.Generator().manual_seed(seed + 7919)
        applied = []
        for v in views:
            d, a = degrade_view(
                torch.as_tensor(v.depth, dtype=torch.float32),
                sigma_rel=knobs["sigma_rel"], scale=knobs["scale"],
                shift=knobs["shift"], scale_jitter=knobs["scale_jitter"],
                shift_jitter=knobs["shift_jitter"], corr_px=knobs["noise_corr_px"],
                generator=g_noise, jitter_generator=g_jitter)
            v.depth = d.numpy()
            applied.append(a)
        info["applied_scales"] = [a["scale"] for a in applied]
        info["applied_shifts"] = [a["shift"] for a in applied]
    else:
        # A real monocular model: relative inverse depth, no metric scale.
        from src.depth_model import (CachedPredictor, align_inverse_depth,
                                     align_inverse_to_points, disparity_to_depth)
        from src.sfm import sparse_depths

        align = dcfg.get("align", "gt")
        info["align"] = align
        if align == "none" and needs_metric:
            raise SystemExit(
                f"depth.model={model_name!r} with align=none yields RELATIVE depth, "
                "but this run needs metric depth\n(init=depth backprojects, or "
                "depth_kind=absolute compares against metres).\n"
                "Use align: gt (oracle) or align: sfm (against triangulated points).")
        predict = CachedPredictor(
            cache_dir or resolve_data_root() / "cache" / "dav2",
            variant=dcfg.get("variant", "large"), local_dir=dcfg.get("local_dir"),
            device=device)
        solved, n_align = [], []
        for k, (v, c) in enumerate(zip(views, clean)):
            disp = predict(v)
            if align == "gt":
                if c is None:
                    raise SystemExit("depth.align=gt needs GT depth on the input "
                                     "views; use align: sfm where there is none.")
                depth, s, t = align_inverse_depth(disp, c, mask=v.mask)
            elif align == "sfm":
                if sfm is None:
                    raise SystemExit("depth.align=sfm but no SfM points were built")
                pts, _, st = sfm
                obs = st.get("_obs")
                # Points this view helped triangulate are the safest targets --
                # none can be occluded here. But 3-view SfM leaves only 2-4 of
                # them per view, so fall back to every point that lands in
                # frame rather than give up on the realistic operating point.
                seen = pts if obs is None else pts[(obs == k).any(axis=1)]
                px, z = sparse_depths(seen, v)
                if len(z) < SFM_ALIGN_PREFERRED:
                    px, z = sparse_depths(pts, v)
                n_align.append(len(z))
                depth, s, t = align_inverse_to_points(disp, px, z)
            elif align == "none":
                depth, s, t = disparity_to_depth(disp).astype(np.float32), 1.0, 0.0
            else:
                raise ValueError(f"unknown depth.align {align!r}; gt | sfm | none")
            if v.mask is not None:
                depth = np.where(v.mask, depth, 0.0).astype(np.float32)
            v.depth = depth
            solved.append((s, t))
        # Where the real model's alignment landed, per view -- log, never discard.
        info["aligned_scale"] = float(np.mean([s for s, _ in solved]))
        info["aligned_shift"] = float(np.mean([t for _, t in solved]))
        if n_align:
            info["align_points"] = float(np.mean(n_align))
            info["align_points_min"] = int(min(n_align))
        info["cache_hits"], info["cache_misses"] = predict.hits, predict.misses
        del predict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {**info, **prior_error_stats([v.depth for v in views], clean,
                                        [v.mask for v in views])}


def scene_scale(views):
    """1.1 x radius of the camera rig, as vanilla 3DGS computes its extent.

    Sets the means learning rate and the densification strategy's scale.
    """
    cams = np.stack([np.asarray(v.c2w)[:3, 3] for v in views])
    r = float(np.linalg.norm(cams - cams.mean(0), axis=1).max())
    return 1.1 * r if r > 0 else 1.0


def build_optimizers(gaussians, scale):
    lrs = dict(LR, means=LR["means"] * scale)
    return {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15)
            for k, p in gaussians.params.items()}


@torch.no_grad()
def _probe_psnr(gaussians, views, device, rasterize, background, sh_degree):
    vals = []
    for v in views:
        rgb = render(gaussians, [v], device, rasterize, background=background,
                     sh_degree=sh_degree)[0][0].clamp(0, 1)
        vals.append(psnr(rgb, torch.as_tensor(v.image, dtype=torch.float32,
                                              device=rgb.device)))
    return float(np.mean(vals))


# --------------------------------------------------------------------------
def train(cfg, gaussians, views, device, out_dir, rasterize=None, strategy=None,
          probe_views=None):
    """rasterize/strategy are injectable so the loop is testable without CUDA."""
    tcfg = cfg.get("train") or {}
    lcfg = cfg.get("loss") or {}
    iters = int(tcfg.get("iterations", 7000))
    lam = float(lcfg.get("depth_lambda", 0.0))
    kind = lcfg.get("depth_kind", "ssi")
    rgb_w = float(lcfg.get("rgb_weight", 1.0))
    alpha_w = float(lcfg.get("alpha_weight", 0.0))
    bg = background_of(cfg)
    scale = scene_scale(views)

    optims = build_optimizers(gaussians, scale)
    # Vanilla 3DGS decays over position_lr_max_steps (30k) whatever the run
    # length; means_lr_steps reproduces that, and defaults to the run itself.
    lr_steps = int(tcfg.get("means_lr_steps") or iters)
    means_decay = torch.optim.lr_scheduler.ExponentialLR(
        optims["means"], gamma=MEANS_LR_FINAL ** (1.0 / max(lr_steps, 1)))
    densify_from = int(tcfg.get("densify_from", 500))
    # A FRACTION of the budget: gsplat's default of 15k inside a 7k run would
    # still be densifying at evaluation time.
    refine_stop = int(float(tcfg.get("densify_until", 0.5)) * iters)
    reset_every = int(tcfg.get("reset_every", 3000))
    if strategy is None:
        from gsplat import DefaultStrategy
        # Opacity reset is switched off INSIDE gsplat and done below instead.
        # gsplat 1.5.3 tests `step % reset_every == 0 & step > 0`, which Python
        # parses as `step % reset_every == (0 & step) > 0` -- always False --
        # so it never resets opacity and random-init floaters are never cleared.
        strategy = DefaultStrategy(
            refine_start_iter=densify_from, refine_stop_iter=refine_stop,
            refine_every=int(tcfg.get("densify_every", 100)),
            reset_every=10 ** 9, verbose=False)
        strategy.check_sanity(gaussians.params, optims)
    state = strategy.initialize_state(scene_scale=scale)

    # Keep targets on the GPU unless they would crowd out the model (the
    # 400-view reference reconstruction in make_gt_depth.py).
    on_device = sum(v.image.nbytes for v in views) < 2e9
    tdev = device if on_device else "cpu"
    gts, masks, gt_depths, depth_masks = [], [], [], []
    for v in views:
        gts.append(torch.as_tensor(v.image, dtype=torch.float32, device=tdev))
        m = None if v.mask is None else torch.as_tensor(v.mask, device=tdev)
        masks.append(m)
        if v.depth is not None and lam > 0:
            d = torch.as_tensor(v.depth, dtype=torch.float32, device=tdev)
            dm = torch.isfinite(d) & (d > 0)
            if m is not None:
                dm &= m
            ok = int(dm.sum()) >= 16
            gt_depths.append(d if ok else None)
            depth_masks.append(dm if ok else None)
        else:
            gt_depths.append(None)
            depth_masks.append(None)

    def dev(x):
        return None if x is None else x.to(device, non_blocking=True)

    probe = list(probe_views or [])[:PROBE_VIEWS]
    probe_every = int(tcfg.get("probe_every", 500))
    rng = np.random.default_rng(tcfg.get("seed", 0))
    log, curve, resets, t0 = [], [], [], time.time()
    for step in range(iters):
        sh_now = min(step // SH_UP_EVERY, gaussians.sh_degree)
        i = int(rng.integers(len(views)))
        rgb, depth, alpha, info = render(gaussians, [views[i]], device, rasterize,
                                         background=bg, sh_degree=sh_now)
        rgb, depth, alpha = rgb[0], depth[0], alpha[0, ..., 0]
        gt = dev(gts[i])

        l1 = (rgb - gt).abs().mean()
        loss_rgb = (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1 - ssim_tensor(rgb, gt))
        loss = rgb_w * loss_rgb
        loss_d = torch.zeros((), device=rgb.device)
        if lam > 0 and gt_depths[i] is not None:
            loss_d = depth_loss(depth, dev(gt_depths[i]), kind=kind,
                                mask=dev(depth_masks[i]))
            loss = loss + lam * loss_d
        if alpha_w > 0 and masks[i] is not None:
            loss = loss + alpha_w * (alpha - dev(masks[i]).float()).abs().mean()

        strategy.step_pre_backward(gaussians.params, optims, state, step, info)
        loss.backward()
        for o in optims.values():
            o.step()
            o.zero_grad(set_to_none=True)
        means_decay.step()
        strategy.step_post_backward(gaussians.params, optims, state, step, info,
                                    packed=PACKED)
        # Vanilla 3DGS: reset every reset_every steps while densifying, plus
        # once at densify_from on a white background (its Blender setting).
        if 0 < step < refine_stop and (step % reset_every == 0
                                       or (bg == 1.0 and step == densify_from)):
            reset_opacity(gaussians.params, optims["opacities"])
            resets.append(step)

        if step % 500 == 0 or step == iters - 1:
            row = {"step": step, "loss": loss.detach().item(),
                   "rgb": loss_rgb.detach().item(), "depth": loss_d.detach().item(),
                   "n_gaussians": len(gaussians)}
            log.append(row)
            print(f"  {step:6d}  loss {row['loss']:.4f}  rgb {row['rgb']:.4f}"
                  f"  depth {row['depth']:.4f}  N={len(gaussians)}", flush=True)
        if probe and ((step + 1) % probe_every == 0 or step == iters - 1):
            curve.append([step + 1, _probe_psnr(gaussians, probe, device, rasterize,
                                                bg, sh_now)])
    return {"train_seconds": time.time() - t0, "log": log, "psnr_curve": curve,
            "scene_scale": scale, "n_gaussians": len(gaussians),
            "means_lr_final": optims["means"].param_groups[0]["lr"],
            "opacity_resets": resets}


# --------------------------------------------------------------------------
def _save_depth_png(path, depth, valid):
    import imageio.v2 as imageio
    import matplotlib

    d = np.where(valid, depth, np.nan)
    lo, hi = (np.nanpercentile(d, 1), np.nanpercentile(d, 99)) if valid.any() else (0, 1)
    x = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = matplotlib.colormaps["viridis"](np.nan_to_num(x))[..., :3]
    rgb[~valid] = 0
    imageio.imwrite(path, (rgb * 255).astype(np.uint8))


def _geometry(gaussians, views, rdepths, rmasks, gt_idx, gdepths, gmasks,
              scene_info, max_cloud, tau_rel):
    a = gaussians.activated
    means = a["means"].detach().cpu().numpy().astype(np.float64)
    opac = a["opacities"].detach().cpu().numpy()

    dtu = scene_info.get("dtu_eval")
    if dtu is not None:
        # Official DTU protocol, in the scan's millimetre frame.
        S = np.asarray(scene_info["scale_mat"], dtype=np.float64)

        def to_mm(p):
            return p @ S[:3, :3].T + S[:3, 3]

        pred = to_mm(fuse_depth_maps(rdepths, views, rmasks))
        out = dtu_chamfer(pred, dtu["stl"], dtu["obs_mask"], dtu["bb"], dtu["res"],
                          dtu["plane"])
        stl = dtu["stl"]
        diag = float(np.linalg.norm(stl.max(0) - stl.min(0)))
        out.update(floater_stats(to_mm(means), opac, subsample(stl, 500_000),
                                 tau=tau_rel * diag))
        return out
    if gt_idx:
        # Both clouds fused from the SAME held-out views, so Chamfer compares
        # like with like: only surfaces those views can see.
        sel = [views[i] for i in gt_idx]
        pred = fuse_depth_maps([rdepths[i] for i in gt_idx], sel,
                               [rmasks[i] for i in gt_idx])
        gt = fuse_depth_maps(gdepths, sel, gmasks)
        pred, gt = subsample(pred, max_cloud, 0), subsample(gt, max_cloud, 1)
        out = chamfer_components(pred, gt)
        if len(gt):
            diag = float(np.linalg.norm(gt.max(0) - gt.min(0)))
            out.update(floater_stats(means, opac, gt, tau=tau_rel * diag))
        return out
    return {}


@torch.no_grad()
def evaluate(gaussians, views, device, out_dir, save_n=3, rasterize=None,
             background=0.0, scene_info=None, max_cloud=300_000, floater_tau_rel=0.02):
    import imageio.v2 as imageio

    out_dir = Path(out_dir)
    scene_info = scene_info or {}
    preds, targets, dep_rows = [], [], []
    rdepths, rmasks, gt_idx, gdepths, gmasks = [], [], [], [], []
    lpips_fn = LPIPS(device=device)
    for j, v in enumerate(views):
        rgb, depth, alpha, _ = render(gaussians, [v], device, rasterize,
                                      background=background)
        rgb, depth, alpha = rgb[0].clamp(0, 1), depth[0], alpha[0, ..., 0]
        preds.append(rgb.cpu())
        targets.append(torch.as_tensor(v.image, dtype=torch.float32))
        d = depth.cpu().numpy().astype(np.float64)
        rvalid = (alpha.cpu().numpy() > 0.5) & np.isfinite(d) & (d > 0)
        rdepths.append(d)
        rmasks.append(rvalid)
        if v.depth is not None:
            gv = np.isfinite(v.depth) & (v.depth > 0)
            if v.mask is not None:
                gv &= v.mask
            if gv.sum() >= 2:
                # UNALIGNED is primary: aligning erases exactly the affine bias
                # Axis C injects. See tests/test_depth_semantics.py.
                dep_rows.append({
                    **{f"unaligned_{k}": val for k, val in
                       depth_metrics(d, v.depth, mask=gv).items()},
                    **{f"aligned_{k}": val for k, val in
                       depth_metrics(d, v.depth, mask=gv, align=True).items()}})
                gt_idx.append(j)
                gdepths.append(v.depth)
                gmasks.append(gv)
        if j < save_n:
            imageio.imwrite(out_dir / f"render_{j:02d}.png", to_image(rgb))
            _save_depth_png(out_dir / f"depth_{j:02d}.png", d, rvalid)

    out = appearance_metrics(preds, targets, lpips_fn)
    if scene_info.get("dataset") == "dtu" and all(v.mask is not None for v in views):
        # The DTU sparse-view protocol scores object pixels only.
        ms = [torch.as_tensor(v.mask[..., None], dtype=torch.float32) for v in views]
        mm = appearance_metrics([p * m for p, m in zip(preds, ms)],
                                [t * m for t, m in zip(targets, ms)], lpips_fn)
        out.update({f"masked_{k}": val for k, val in mm.items()})
    if dep_rows:
        for k in dep_rows[0]:
            out[k] = float(np.nanmean([r[k] for r in dep_rows]))
    out.update(_geometry(gaussians, views, rdepths, rmasks, gt_idx, gdepths, gmasks,
                         scene_info, max_cloud, floater_tau_rel))
    return out


# --------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def write_json_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    os.replace(tmp, path)


def run(cfg, out, save_n=3):
    tcfg = cfg.get("train") or {}
    seed = int(tcfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU: gsplat cannot rasterize here.")
    device = torch.device("cuda")
    t0 = time.time()

    print(f"[{cfg.get('dataset', 'blender')}/{cfg.get('scene')}] init={cfg.get('init')} "
          f"n_views={cfg.get('n_views')} "
          f"lambda={(cfg.get('loss') or {}).get('depth_lambda')}", flush=True)
    train_views, eval_views, info = load_scene_and_split(cfg)
    sfm = triangulate_if_needed(cfg, train_views, info)

    dcfg = cfg.get("depth") or {}
    cache_dir = (resolve_data_root() / "cache" / f"dav2_{dcfg.get('variant', 'large')}"
                 / f"{cfg.get('dataset', 'blender')}_{cfg.get('scene')}"
                   f"_ds{cfg.get('downscale', 1)}")
    depth_info = attach_depth(cfg, train_views, device, sfm=sfm, cache_dir=cache_dir)

    gaussians, init_info = build_initial_gaussians(cfg, train_views, device=device,
                                                   sfm=sfm)
    print(f"  initialized {len(gaussians)} gaussians ({cfg.get('init')})", flush=True)

    stats = train(cfg, gaussians, train_views, device, out, probe_views=eval_views)
    metrics = evaluate(gaussians, eval_views, device, out, save_n=save_n,
                       background=background_of(cfg), scene_info=info)
    if cfg.get("save_ply", False):
        gaussians.save_ply(out / "points.ply")

    prior = {k: v for k, v in depth_info.items() if k.startswith("prior_")}
    record = {"config": cfg,
              "depth": {k: v for k, v in depth_info.items() if not k.startswith("prior_")},
              **prior, "init": init_info,
              "train_ids": info["train_ids"], "eval_ids": info["eval_ids"],
              "n_train_views": len(train_views), "n_eval_views": len(eval_views),
              **{k: v for k, v in stats.items() if k != "log"}, **metrics,
              "wall_seconds": time.time() - t0}
    write_json_atomic(out / "train_log.json", stats["log"])
    write_json_atomic(out / "metrics.json", record)      # last: the resume key
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                           if isinstance(v, float)), flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--save-renders", type=int, default=3,
                    help="how many held-out renders (+ depth maps) to save")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(cfg, out, args.save_renders)
    except (Exception, SystemExit) as e:
        summary = " ".join(str(e).split()) or type(e).__name__
        (out / "error.txt").write_text(
            traceback.format_exc() + f"\nERROR: {type(e).__name__}: {summary}\n")
        raise


if __name__ == "__main__":
    main()
