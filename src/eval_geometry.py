"""Geometry and depth evaluation metrics -- the geometry half of the hypothesis.

Three families, each answering a different question:

  depth_metrics   per-pixel depth error on held-out views vs GT depth. Reported
                  UNALIGNED (primary: aligning erases the affine bias Axis C
                  injects) and aligned (secondary).
  chamfer_*       3D surface error between a cloud fused from rendered depth
                  and the GT surface. dtu_chamfer follows the official DTU
                  protocol (ObsMask, ground plane, 20 mm cap), as ported by
                  DTUeval-python, so DTU numbers are comparable to the literature.
  floater_stats   opaque Gaussians far from any GT surface -- the failure a
                  wrong prior produces that depth maps at eval views can miss.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

_NAN = float("nan")


def _tree(points):
    if cKDTree is None:
        raise ImportError("scipy is required for geometry metrics")
    # balanced_tree/compact_nodes off: building a balanced tree over a few
    # million scattered points costs far more than the query time it saves.
    return cKDTree(np.asarray(points, dtype=np.float64), balanced_tree=False,
                   compact_nodes=False)


def _query(tree, points):
    """Nearest-neighbour distances, using every core the host has."""
    return tree.query(np.asarray(points, dtype=np.float64), workers=-1)[0]


def chamfer_distance(pred_points, gt_points):
    """Symmetric Chamfer as the SUM of the two mean nearest distances."""
    pred, gt = np.asarray(pred_points), np.asarray(gt_points)
    d_pred = _query(_tree(gt), pred)
    d_gt = _query(_tree(pred), gt)
    return float(d_pred.mean() + d_gt.mean())


def chamfer_components(pred_points, gt_points, max_dist=None):
    """Accuracy (pred->GT), completeness (GT->pred), and their MEAN (DTU style).

    Distances >= max_dist are discarded before averaging when it is given, as
    the DTU protocol does with its 20 mm cap.
    """
    pred, gt = np.asarray(pred_points), np.asarray(gt_points)
    if len(pred) == 0 or len(gt) == 0:
        return {"chamfer_acc": _NAN, "chamfer_comp": _NAN, "chamfer": _NAN}
    d_acc = _query(_tree(gt), pred)
    d_comp = _query(_tree(pred), gt)
    if max_dist is not None:
        d_acc, d_comp = d_acc[d_acc < max_dist], d_comp[d_comp < max_dist]
    acc = float(d_acc.mean()) if len(d_acc) else _NAN
    comp = float(d_comp.mean()) if len(d_comp) else _NAN
    return {"chamfer_acc": acc, "chamfer_comp": comp, "chamfer": (acc + comp) / 2}


def depth_metrics(pred, gt, mask=None, align=False, eps=1e-6):
    """MAE / RMSE / AbsRel / delta<1.25, with optional scale+shift alignment.

    Only pixels with finite, positive GT and finite prediction are scored;
    `mask` further restricts them (it never re-admits invalid GT).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if valid.sum() < 2:
        return {"mae": _NAN, "rmse": _NAN, "absrel": _NAN, "delta1": _NAN}
    p, g = pred[valid], gt[valid]
    if align:
        A = np.stack([p, np.ones_like(p)], axis=1)
        s, t = np.linalg.lstsq(A, g, rcond=None)[0]
        p = s * p + t
    diff = p - g
    ratio = np.maximum(p / (g + eps), g / (np.abs(p) + eps))
    return {
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "absrel": float((np.abs(diff) / (g + eps)).mean()),
        "delta1": float(((ratio < 1.25) & (p > 0)).mean()),
    }


def fuse_depth_maps(depths, views, masks=None, stride=1):
    """Backproject several depth maps into one world-space cloud."""
    from src.depth_init import backproject_depth

    pts = []
    for i, (d, v) in enumerate(zip(depths, views)):
        d = np.asarray(d, dtype=np.float64).copy()
        if masks is not None and masks[i] is not None:
            d[~np.asarray(masks[i], dtype=bool)] = 0.0
        pts.append(backproject_depth(d, v.K, v.c2w, stride=stride))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def subsample(points, n, seed=0):
    points = np.asarray(points)
    if len(points) <= n:
        return points
    return points[np.random.default_rng(seed).choice(len(points), n, replace=False)]


def voxel_downsample(points, voxel):
    """Keep one point per occupied voxel (the first seen).

    Voxel indices are folded into a single int64 key so the dedupe sorts a
    flat array instead of 3-column records: np.unique(..., axis=0) on a few
    million rows takes minutes, and a run with bad geometry fuses into exactly
    such a cloud. Falls back to the record sort if the grid is too large to
    index (only possible for a cloud spread over a huge volume).
    """
    points = np.asarray(points)
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) < 2.0 ** 62:
        flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
        _, idx = np.unique(flat, return_index=True)
    else:                                           # pragma: no cover
        _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def floater_stats(means, opacities, surface_points, tau, min_opacity=0.5):
    """Opaque Gaussians whose centre is farther than `tau` from the GT surface."""
    means = np.asarray(means)
    sel = np.asarray(opacities) > min_opacity
    n_opaque = int(sel.sum())
    if n_opaque == 0 or len(surface_points) == 0:
        return {"floater_frac": _NAN, "floater_count": 0, "n_opaque": n_opaque}
    d = _query(_tree(surface_points), means[sel])
    far = d > tau
    return {"floater_frac": float(far.mean()), "floater_count": int(far.sum()),
            "n_opaque": n_opaque}


def reprojection_consistency(depth_a, view_a, depth_b, view_b, occl_tol=0.05,
                             stride=2):
    """How much two depth maps of the same surface disagree.

    Backproject view a's depth, project it into view b, and compare the
    point's depth in b with b's depth map at the landing pixel. Points more
    than occl_tol (relative) BEHIND b's surface are occluded in b and skipped;
    points in front of it are genuine disagreement and kept.

    Returns {"median_rel", "within_1pct", "n"}. Used to quantify the error of
    the NeRF-Synthetic reference depth itself (scripts/make_gt_depth.py).
    """
    from src.depth_init import backproject_depth

    pts = backproject_depth(np.asarray(depth_a, dtype=np.float64), view_a.K,
                            view_a.c2w, stride=stride)
    w2c = np.linalg.inv(np.asarray(view_b.c2w, dtype=np.float64))
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    front = z > 1e-9
    pc, z = pc[front], z[front]
    K = view_b.K
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    db_map = np.asarray(depth_b, dtype=np.float64)
    H, W = db_map.shape
    j, i = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inb = (j >= 0) & (j < W) & (i >= 0) & (i < H)
    db = db_map[i[inb], j[inb]]
    zin = z[inb]
    ok = np.isfinite(db) & (db > 0)
    rel = (zin[ok] - db[ok]) / db[ok]
    rel = np.abs(rel[rel < occl_tol])
    if len(rel) == 0:
        return {"median_rel": _NAN, "within_1pct": _NAN, "n": 0}
    return {"median_rel": float(np.median(rel)),
            "within_1pct": float((rel < 0.01).mean()), "n": int(len(rel))}


def dtu_chamfer(pred_mm, stl_mm, obs_mask, bb, res, plane, voxel=0.2,
                max_dist=20.0, patch=60.0):
    """Official DTU point-cloud evaluation, after DTUeval-python.

    pred_mm, stl_mm: (N, 3) clouds in the DTU scan frame, millimetres.
    obs_mask: (X, Y, Z) bool volume; bb: (2, 3) its bounds; res: voxel size (mm).
    plane: (4,) ground plane; completeness counts only GT points above it.

    Accuracy uses prediction points inside the observed volume; completeness
    uses GT points above the plane against all in-bound prediction points.
    Both discard distances >= max_dist, and 'chamfer' is their mean.
    """
    pred = np.asarray(pred_mm, dtype=np.float64)
    bb = np.asarray(bb, dtype=np.float64)
    inbound = ((pred >= bb[0] - patch) & (pred < bb[1] + patch * 2)).all(axis=1)
    data_in = voxel_downsample(pred[inbound], voxel)
    grid = np.around((data_in - bb[0]) / res).astype(np.int64)
    grid_in = ((grid >= 0) & (grid < np.array(obs_mask.shape))).all(axis=1)
    g = grid[grid_in]
    in_obs = np.asarray(obs_mask)[g[:, 0], g[:, 1], g[:, 2]].astype(bool)
    data_in_obs = data_in[grid_in][in_obs]

    stl = np.asarray(stl_mm, dtype=np.float64)
    above = (np.concatenate([stl, np.ones((len(stl), 1))], axis=1)
             @ np.asarray(plane, dtype=np.float64).reshape(4)) > 0
    stl_above = stl[above]

    out = {"n_eval_points": int(len(data_in_obs))}
    if len(data_in_obs) == 0 or len(data_in) == 0:
        return {**out, "chamfer_acc": _NAN, "chamfer_comp": _NAN, "chamfer": _NAN}
    d_acc = _query(_tree(stl), data_in_obs)
    d_comp = _query(_tree(data_in), stl_above)
    acc = float(d_acc[d_acc < max_dist].mean()) if (d_acc < max_dist).any() else _NAN
    comp = float(d_comp[d_comp < max_dist].mean()) if (d_comp < max_dist).any() else _NAN
    return {**out, "chamfer_acc": acc, "chamfer_comp": comp, "chamfer": (acc + comp) / 2}
