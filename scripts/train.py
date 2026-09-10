"""Train one depth-prior 3DGS run from a config, and write metrics.json.

The contract with scripts/run_sweep.py: read --config, write metrics.json into
--out. run_sweep treats metrics.json as the resume key, so it must be written
only on success.

Every run reports BOTH metric families. A run with appearance numbers only
cannot contribute to the appearance-vs-geometry map, which is the contribution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.corruptions import degrade                                # noqa: E402
from src.data import (blender_sparse_split, evenly_spaced,         # noqa: E402
                      load_blender_scene, load_dtu_scene, dtu_sparse_split)
from src.depth_loss import depth_loss                              # noqa: E402
from src.eval_geometry import depth_metrics                        # noqa: E402
from src.gaussians import build_initial_gaussians                  # noqa: E402
from src.metrics import LPIPS, appearance_metrics, ssim_tensor     # noqa: E402
from src.render import render, to_image                            # noqa: E402

# 3DGS defaults. means lr is scaled by scene extent; the rest are absolute.
LR = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3,
      "opacities": 5e-2, "colors": 2.5e-3}
SSIM_WEIGHT = 0.2       # L_rgb = (1-w)*L1 + w*(1-SSIM), as in vanilla 3DGS


# --------------------------------------------------------------------------
def load_scene_and_split(cfg):
    """Return (train_views, eval_views, scene_meta)."""
    ds = cfg.get("dataset", "blender")
    n = cfg.get("n_views", 3)
    if ds == "blender":
        tr = load_blender_scene(scene=cfg["scene"], split="train")
        te = load_blender_scene(scene=cfg["scene"], split="test")
        ids = blender_sparse_split(n, len(tr))
        eval_ids = evenly_spaced(cfg.get("eval_views", 16), len(te))
        return [tr.views[i] for i in ids], [te.views[i] for i in eval_ids], tr.meta
    if ds == "dtu":
        sc = load_dtu_scene(scan_id=cfg["scene"])
        ids = dtu_sparse_split(n)
        held = [i for i in range(len(sc)) if i not in ids]
        eval_ids = [held[i] for i in evenly_spaced(cfg.get("eval_views", 16), len(held))]
        return [sc.views[i] for i in ids], [sc.views[i] for i in eval_ids], sc.meta
    raise ValueError(f"unknown dataset {ds!r}")


def attach_depth(cfg, views, device):
    """Put the (possibly corrupted) prior depth on each training view.

    Returns a dict describing what was done, which goes into metrics.json --
    a degradation curve is unreadable without knowing what was actually applied.
    """
    dcfg = cfg.get("depth", {})
    model_name = dcfg.get("model", "gt")
    uses_init = cfg.get("init") == "depth"
    uses_loss = cfg.get("loss", {}).get("depth_lambda", 0.0) > 0
    needs_metric = uses_init or \
        (uses_loss and cfg.get("loss", {}).get("depth_kind") == "absolute")
    info = {"model": model_name, "sigma_rel": dcfg.get("sigma_rel", 0.0),
            "scale": dcfg.get("scale", 1.0), "shift": dcfg.get("shift", 0.0)}

    # The prior reaches a run through exactly two paths: initialization and the
    # depth loss. With neither active there is nothing to prepare, and demanding
    # depth would block the runs that need none -- the same reasoning that
    # prunes cells in src/sweep.py.
    if not uses_init and not uses_loss:
        return {**info, "used": False}
    info["used"] = True

    if model_name == "gt":
        if any(v.depth is None for v in views):
            raise SystemExit(
                "depth.model is 'gt' but the views carry no depth.\n"
                "NeRF-Synthetic ships no metric depth: render it from the "
                "original .blend files into\n"
                "  $DATA_ROOT/nerf_synthetic/<scene>/depth_train/r_<i>.npy\n"
                "Runs with init=random and depth_lambda=0 need no depth and "
                "work today.")
        gen = torch.Generator().manual_seed(cfg.get("train", {}).get("seed", 0))
        for v in views:
            d = torch.as_tensor(v.depth, dtype=torch.float32)
            v.depth = degrade(d, dcfg.get("sigma_rel", 0.0), dcfg.get("scale", 1.0),
                              dcfg.get("shift", 0.0), generator=gen).numpy()
        return info

    # A real monocular model: relative inverse depth, no metric scale.
    from src.depth_model import load_depth_model, predict_disparity, \
        disparity_to_depth, align_to_metric
    model, proc, dev = load_depth_model(variant=dcfg.get("variant", "large"),
                                        local_dir=dcfg.get("local_dir"), device=device)
    solved = []
    for v in views:
        rel = disparity_to_depth(predict_disparity(model, proc, v.image, dev))
        if needs_metric:
            if v.depth is None:
                raise SystemExit(
                    f"depth.model={model_name!r} yields RELATIVE depth, but this "
                    "run needs metric depth\n(init=depth backprojects, or "
                    "depth_kind=absolute compares against metres).\n"
                    "Provide GT depth to align against, or use a "
                    "scale-invariant depth_kind with init=random.")
            rel, s, t = align_to_metric(rel, v.depth, mask=v.mask)
            solved.append((s, t))
        v.depth = np.asarray(rel, dtype=np.float32)
    if solved:
        # Where the real model sits relative to the synthetic Axis C curve.
        info["aligned_scale"] = float(np.mean([s for s, _ in solved]))
        info["aligned_shift"] = float(np.mean([t for _, t in solved]))
    return info


def scene_scale(views):
    """Radius of the camera rig -- sets the means learning rate and strategy scale."""
    cams = np.stack([np.asarray(v.c2w)[:3, 3] for v in views])
    return float(np.linalg.norm(cams - cams.mean(0), axis=1).max()) or 1.0


def build_optimizers(gaussians, scale):
    lrs = dict(LR, means=LR["means"] * scale)
    return {k: torch.optim.Adam([p], lr=lrs[k], eps=1e-15)
            for k, p in gaussians.params.items()}


# --------------------------------------------------------------------------
def train(cfg, gaussians, views, device, out_dir, rasterize=None, strategy=None):
    """rasterize/strategy are injectable so the loop is testable without CUDA."""
    iters = cfg.get("train", {}).get("iterations", 7000)
    lam = cfg.get("loss", {}).get("depth_lambda", 0.0)
    kind = cfg.get("loss", {}).get("depth_kind", "ssi")
    scale = scene_scale(views)

    optims = build_optimizers(gaussians, scale)
    if strategy is None:
        from gsplat import DefaultStrategy
        strategy = DefaultStrategy(verbose=False)
        strategy.check_sanity(gaussians.params, optims)
    state = strategy.initialize_state(scene_scale=scale)

    gts = [torch.as_tensor(v.image, dtype=torch.float32, device=device) for v in views]
    gt_depths, depth_masks = [], []
    for v in views:
        if v.depth is not None and lam > 0:
            d = torch.as_tensor(v.depth, dtype=torch.float32, device=device)
            m = torch.as_tensor(v.mask, device=device) if v.mask is not None \
                else torch.ones_like(d, dtype=torch.bool)
            gt_depths.append(d)
            depth_masks.append(m & torch.isfinite(d) & (d > 0))
        else:
            gt_depths.append(None)
            depth_masks.append(None)

    rng = np.random.default_rng(cfg.get("train", {}).get("seed", 0))
    log, t0 = [], time.time()
    for step in range(iters):
        i = int(rng.integers(len(views)))
        rgb, depth, _, info = render(gaussians, [views[i]], device, rasterize)
        rgb, depth = rgb[0], depth[0]

        l1 = (rgb - gts[i]).abs().mean()
        loss_rgb = (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (1 - ssim_tensor(rgb, gts[i]))
        loss = loss_rgb
        loss_d = torch.zeros((), device=device)
        if lam > 0 and gt_depths[i] is not None:
            loss_d = depth_loss(depth, gt_depths[i], kind=kind, mask=depth_masks[i])
            loss = loss + lam * loss_d

        strategy.step_pre_backward(gaussians.params, optims, state, step, info)
        loss.backward()
        for o in optims.values():
            o.step()
            o.zero_grad(set_to_none=True)
        strategy.step_post_backward(gaussians.params, optims, state, step, info)

        if step % 500 == 0 or step == iters - 1:
            row = {"step": step, "loss": loss.detach().item(),
                   "rgb": loss_rgb.detach().item(), "depth": loss_d.detach().item(),
                   "n_gaussians": len(gaussians)}
            log.append(row)
            print(f"  {step:6d}  loss {row['loss']:.4f}  rgb {row['rgb']:.4f}"
                  f"  depth {row['depth']:.4f}  N={len(gaussians)}")
    return {"train_seconds": time.time() - t0, "log": log,
            "scene_scale": scale, "n_gaussians": len(gaussians)}


@torch.no_grad()
def evaluate(gaussians, views, device, out_dir, save_n=3, rasterize=None):
    preds, targets, dep_rows = [], [], []
    lpips_fn = LPIPS(device=device)
    for j, v in enumerate(views):
        rgb, depth, _, _ = render(gaussians, [v], device, rasterize)
        rgb, depth = rgb[0], depth[0]
        preds.append(rgb.clamp(0, 1).cpu())
        targets.append(torch.as_tensor(v.image, dtype=torch.float32))
        if v.depth is not None:
            m = v.mask if v.mask is not None else None
            d = depth.cpu().numpy()
            # UNALIGNED is primary: aligning erases exactly the affine bias
            # Axis C injects. See tests/test_depth_semantics.py.
            dep_rows.append({
                **{f"unaligned_{k}": val for k, val in
                   depth_metrics(d, v.depth, mask=m, align=False).items()},
                **{f"aligned_{k}": val for k, val in
                   depth_metrics(d, v.depth, mask=m, align=True).items()}})
        if j < save_n:
            import imageio.v2 as imageio
            imageio.imwrite(out_dir / f"render_{j:02d}.png", to_image(rgb))

    out = appearance_metrics(preds, targets, lpips_fn)
    if dep_rows:
        for k in dep_rows[0]:
            out[k] = float(np.mean([r[k] for r in dep_rows]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/demo")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.get("train", {}).get("seed", 0))
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU: gsplat cannot rasterize here.")
    device = torch.device("cuda")

    print(f"[{cfg.get('scene')}] init={cfg.get('init')} n_views={cfg.get('n_views')} "
          f"lambda={cfg.get('loss', {}).get('depth_lambda')}")
    train_views, eval_views, _ = load_scene_and_split(cfg)
    depth_info = attach_depth(cfg, train_views, device)

    gaussians = build_initial_gaussians(cfg, train_views, device=device)
    print(f"  initialized {len(gaussians)} gaussians ({cfg.get('init')})")

    stats = train(cfg, gaussians, train_views, device, out)
    metrics = evaluate(gaussians, eval_views, device, out)
    gaussians.save_ply(out / "points.ply")

    record = {"config": cfg, "depth": depth_info,
              "n_train_views": len(train_views), "n_eval_views": len(eval_views),
              **{k: v for k, v in stats.items() if k != "log"}, **metrics}
    json.dump(record, open(out / "metrics.json", "w"), indent=2)
    json.dump(stats["log"], open(out / "train_log.json", "w"), indent=2)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                           if isinstance(v, float)))


if __name__ == "__main__":
    main()
