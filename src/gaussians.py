"""The Gaussian parameter set and the three initializations (Axis A).

Deliberately free of any gsplat/CUDA dependency: initialization is half of what
this study measures, so it must be inspectable and testable on any machine.
The rasterizer is isolated in src/render.py.

Parameter conventions follow what gsplat's strategies require: a ParameterDict
with "means", "scales", "quats", "opacities", stored in their raw
(pre-activation) form -- log scales, logit opacities, unnormalized quaternions.
Colour is stored as spherical-harmonic coefficients exactly as vanilla 3DGS
stores it ("sh0" is the DC term; "shN" the higher bands when sh_degree > 0), so
the rasterizer's colour path, including its clamp_min(0), is the reference one.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh):
    return sh * SH_C0 + 0.5


def _inverse_sigmoid(x):
    return float(np.log(x / (1.0 - x)))


RESET_OPACITY = 0.01     # vanilla 3DGS caps every opacity at 0.01 on reset


@torch.no_grad()
def reset_opacity(params, optimizer, value=RESET_OPACITY):
    """Vanilla 3DGS opacity reset: cap every opacity at `value`, clear its Adam state.

    Gaussians the images need regain opacity within a few hundred steps;
    floaters they do not need stay transparent and are pruned. In place, so
    the optimizer keeps pointing at the same parameter.
    """
    p = params["opacities"]
    p.clamp_(max=_inverse_sigmoid(value))
    state = optimizer.state.get(p, {})
    for key in ("exp_avg", "exp_avg_sq"):
        if key in state:
            state[key].zero_()


def knn_scale(points, k=3, floor=1e-7):
    """Initial isotropic scale per point: mean distance to its k nearest others.

    Standard 3DGS initialization. Done in numpy so it stays CPU-testable; the
    point counts here (<=1e6) make an exact KD-tree query affordable.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= k:
        return np.full(len(pts), 0.01)
    # k+1 because the first neighbour of a point is itself.
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    return np.maximum(d[:, 1:].mean(axis=1), floor)


class Gaussians(nn.Module):
    """A trainable Gaussian cloud.

    The study's default is sh_degree=0 (view-independent colour): it varies
    initialization, view count and depth supervision, and higher SH bands let a
    sparse-view model fit training views with view-dependent colour instead of
    geometry -- a confound this project does not control for. sh_degree=3
    reproduces vanilla 3DGS for the reproduction gate.
    """

    def __init__(self, means, colors, scales=None, opacity=0.1, sh_degree=0,
                 device="cpu"):
        super().__init__()
        means = torch.as_tensor(np.asarray(means), dtype=torch.float32)
        colors = torch.as_tensor(np.asarray(colors), dtype=torch.float32)
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(f"means must be (N,3), got {tuple(means.shape)}")
        if len(colors) != len(means):
            raise ValueError(f"{len(colors)} colors for {len(means)} means")
        if not 0 <= sh_degree <= 3:
            raise ValueError(f"sh_degree must be 0..3, got {sh_degree}")

        n = len(means)
        if scales is None:
            scales = knn_scale(means.numpy())
        scales = torch.as_tensor(np.asarray(scales), dtype=torch.float32)
        if scales.ndim == 1:
            scales = scales[:, None].repeat(1, 3)

        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0                       # wxyz identity rotation

        params = {
            "means": nn.Parameter(means),
            "scales": nn.Parameter(torch.log(scales.clamp(min=1e-8))),
            "quats": nn.Parameter(quats),
            "opacities": nn.Parameter(torch.full((n,), _inverse_sigmoid(opacity))),
            "sh0": nn.Parameter(rgb_to_sh(colors.clamp(0.0, 1.0))[:, None, :]),
        }
        if sh_degree > 0:
            params["shN"] = nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3))
        self.params = nn.ParameterDict(params)
        self.sh_degree = sh_degree
        self.to(device)

    def __len__(self):
        return self.params["means"].shape[0]

    # Activations -- raw parameters are unconstrained, these are what the
    # rasterizer consumes.
    @property
    def activated(self):
        p = self.params
        sh = p["sh0"] if "shN" not in p else torch.cat([p["sh0"], p["shN"]], dim=1)
        return dict(
            means=p["means"],
            quats=p["quats"],
            scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]),
            sh=sh,
            rgb=torch.clamp(sh_to_rgb(p["sh0"][:, 0]), 0.0, 1.0),
        )

    def save_ply(self, path):
        """Binary PLY of centres, DC colour and opacity, for figures and viewers."""
        a = self.activated
        xyz = a["means"].detach().cpu().numpy()
        rgb = (a["rgb"].detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)
        op = a["opacities"].detach().cpu().numpy()
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                       ("opacity", "<f4")])
        arr = np.empty(len(xyz), dtype=dt)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        arr["opacity"] = op
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(arr)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "property float opacity\nend_header\n")
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(arr.tobytes())


# --------------------------------------------------------------------------
# Axis A: the three initializations
# --------------------------------------------------------------------------
def _cap(pts, cols, max_points, seed):
    if len(pts) > max_points:
        idx = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols


def init_random(n_points=100_000, extent=1.3, seed=0, sh_degree=0, device="cpu"):
    """Uniform points in a cube, grey colors -- vanilla 3DGS's Blender default.

    NeRF-Synthetic has no SfM cloud, so the original paper initializes it
    randomly. This is the honest baseline for those scenes, not a fallback.
    """
    rng = np.random.default_rng(seed)
    means = rng.uniform(-extent, extent, size=(n_points, 3))
    colors = np.full((n_points, 3), 0.5)
    return Gaussians(means, colors, sh_degree=sh_degree, device=device)


def init_from_depth(views, max_points=200_000, stride=2, seed=0, sh_degree=0,
                    device="cpu"):
    """Backproject each input view's depth into one cloud (Axis A, 'depth').

    Consumes METRIC depth. This is the path through which a scale-biased prior
    reaches geometry -- a scale-and-shift-invariant loss cannot see such a bias,
    but backprojection displaces the cloud bodily. See
    tests/test_depth_semantics.py.
    """
    from src.depth_init import backproject_depth

    all_pts, all_cols = [], []
    for v in views:
        if v.depth is None:
            raise ValueError(f"view {v.name!r} has no depth; cannot use init=depth")
        depth = np.asarray(v.depth, dtype=np.float64).copy()
        if v.mask is not None:
            depth[~v.mask] = 0.0            # drop background rays
        pts, cols = backproject_depth(depth, v.K, v.c2w, rgb=v.image, stride=stride)
        all_pts.append(pts)
        all_cols.append(cols)

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    if len(pts) == 0:
        raise ValueError("depth init produced no points (all depth invalid?)")
    pts, cols = _cap(pts, cols, max_points, seed)
    return Gaussians(pts, cols, sh_degree=sh_degree, device=device)


def init_from_points(points, colors, max_points=200_000, seed=0, sh_degree=0,
                     device="cpu"):
    """Initialize from a sparse SfM cloud (Axis A, 'sfm')."""
    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors, dtype=np.float64)
    pts, cols = _cap(pts, cols, max_points, seed)
    return Gaussians(pts, cols, sh_degree=sh_degree, device=device)


def build_initial_gaussians(cfg, views, device="cpu", sfm=None):
    """Dispatch on cfg['init']. Returns (gaussians, init_info).

    `sfm` is a precomputed (points, colors, stats) from src.sfm.triangulate_views,
    passed in when the caller already triangulated (e.g. to align a real depth
    model against the same points); otherwise it is computed here.
    """
    kind = cfg.get("init", "random")
    io = cfg.get("init_opts") or {}
    tcfg = cfg.get("train") or {}
    seed, sh = tcfg.get("seed", 0), tcfg.get("sh_degree", 0)
    max_points = io.get("max_points", 200_000)

    if kind == "random":
        g = init_random(io.get("random_points", 100_000), io.get("random_extent", 1.3),
                        seed=seed, sh_degree=sh, device=device)
        return g, {"init": kind, "n_init_points": len(g)}
    if kind == "depth":
        g = init_from_depth(views, max_points=max_points,
                            stride=io.get("depth_stride", 2), seed=seed,
                            sh_degree=sh, device=device)
        return g, {"init": kind, "n_init_points": len(g)}
    if kind == "sfm":
        if sfm is None:
            from src.sfm import triangulate_views
            sfm = triangulate_views(views, **(cfg.get("sfm") or {}))
        pts, cols, stats = sfm
        min_pts = io.get("sfm_min_points", 4)
        if len(pts) < min_pts:
            raise ValueError(
                f"SfM triangulated {len(pts)} point(s) from {len(views)} views "
                f"(need >= {min_pts}); init=sfm cannot start. With few views "
                "this is the expected failure of SfM, and is itself a result.")
        g = init_from_points(pts, cols, max_points, seed, sh, device)
        public = {k: v for k, v in stats.items() if not k.startswith("_")}
        return g, {"init": kind, "n_init_points": len(g), "sfm": public}
    raise ValueError(f"unknown init {kind!r}; expected sfm | random | depth")
