"""The Gaussian parameter set and the three initializations (Axis A).

Deliberately free of any gsplat/CUDA dependency: initialization is half of what
this study measures, so it must be inspectable and testable on any machine.
The rasterizer is isolated in src/render.py.

Parameter conventions follow what gsplat's strategies require: a ParameterDict
with "means", "scales", "quats", "opacities", stored in their raw
(pre-activation) form -- log scales, logit opacities, unnormalized quaternions.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _inverse_sigmoid(x):
    return float(np.log(x / (1.0 - x)))


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

    Colors are plain RGB (sh_degree=None at the rasterizer). The study varies
    initialization, view count and depth supervision; adding spherical
    harmonics would introduce a confound this project does not control for,
    and NeRF-Synthetic objects are near-Lambertian.
    """

    def __init__(self, means, colors, scales=None, opacity=0.1, device="cpu"):
        super().__init__()
        means = torch.as_tensor(np.asarray(means), dtype=torch.float32)
        colors = torch.as_tensor(np.asarray(colors), dtype=torch.float32)
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(f"means must be (N,3), got {tuple(means.shape)}")
        if len(colors) != len(means):
            raise ValueError(f"{len(colors)} colors for {len(means)} means")

        n = len(means)
        if scales is None:
            scales = knn_scale(means.numpy())
        scales = torch.as_tensor(np.asarray(scales), dtype=torch.float32)
        if scales.ndim == 1:
            scales = scales[:, None].repeat(1, 3)

        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0                       # wxyz identity rotation

        self.params = nn.ParameterDict({
            "means": nn.Parameter(means),
            "scales": nn.Parameter(torch.log(scales.clamp(min=1e-8))),
            "quats": nn.Parameter(quats),
            "opacities": nn.Parameter(
                torch.full((n,), _inverse_sigmoid(opacity))),
            "colors": nn.Parameter(colors.clamp(0.0, 1.0)),
        })
        self.to(device)

    def __len__(self):
        return self.params["means"].shape[0]

    # Activations -- raw parameters are unconstrained, these are what the
    # rasterizer consumes.
    @property
    def activated(self):
        p = self.params
        return dict(
            means=p["means"],
            quats=p["quats"],
            scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]),
            colors=torch.clamp(p["colors"], 0.0, 1.0),
        )

    def save_ply(self, path):
        """Write positions and colors for the geometry metric and for viewers."""
        a = self.activated
        xyz = a["means"].detach().cpu().numpy()
        rgb = (a["colors"].detach().cpu().numpy() * 255).astype(np.uint8)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(xyz)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x} {y} {z} {r} {g} {b}\n")


# --------------------------------------------------------------------------
# Axis A: the three initializations
# --------------------------------------------------------------------------
def init_random(n_points=100_000, extent=1.3, seed=0, device="cpu"):
    """Uniform points in a cube, grey colors -- vanilla 3DGS's Blender default.

    NeRF-Synthetic has no SfM cloud, so the original paper initializes it
    randomly. This is the honest baseline for those scenes, not a fallback.
    """
    rng = np.random.default_rng(seed)
    means = rng.uniform(-extent, extent, size=(n_points, 3))
    colors = np.full((n_points, 3), 0.5)
    return Gaussians(means, colors, device=device)


def init_from_depth(views, max_points=200_000, stride=2, seed=0, device="cpu"):
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
    if len(pts) > max_points:
        idx = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return Gaussians(pts, cols, device=device)


def init_from_sfm(points_path, max_points=200_000, seed=0, device="cpu"):
    """Initialize from a COLMAP/SfM cloud (Axis A, 'sfm')."""
    from pathlib import Path
    from src.data import _load_ply_points

    p = Path(points_path)
    if not p.exists():
        raise FileNotFoundError(
            f"no SfM point cloud at {p}.\n"
            "NeRF-Synthetic ships none -- COLMAP must be run over the selected\n"
            "input views first. The original 3DGS paper uses RANDOM init for\n"
            "Blender scenes for exactly this reason, so on this dataset\n"
            "init=random is the meaningful baseline, not init=sfm.")
    pts = _load_ply_points(p)
    if len(pts) > max_points:
        idx = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return Gaussians(pts, np.full((len(pts), 3), 0.5), device=device)


def build_initial_gaussians(cfg, views, device="cpu", sfm_path=None):
    """Dispatch on cfg['init']."""
    kind = cfg.get("init", "random")
    seed = cfg.get("train", {}).get("seed", 0)
    if kind == "random":
        return init_random(seed=seed, device=device)
    if kind == "depth":
        return init_from_depth(views, seed=seed, device=device)
    if kind == "sfm":
        return init_from_sfm(sfm_path, seed=seed, device=device)
    raise ValueError(f"unknown init {kind!r}; expected sfm | random | depth")
