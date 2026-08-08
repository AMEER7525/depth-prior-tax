"""Geometry and depth evaluation metrics — the geometry half of the hypothesis."""
from __future__ import annotations
import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


def chamfer_distance(pred_points, gt_points):
    """Symmetric Chamfer distance between (N,3) and (M,3) clouds. Requires scipy."""
    if cKDTree is None:
        raise ImportError("scipy is required for chamfer_distance")
    pred, gt = np.asarray(pred_points), np.asarray(gt_points)
    d_pred, _ = cKDTree(gt).query(pred)
    d_gt, _ = cKDTree(pred).query(gt)
    return float(d_pred.mean() + d_gt.mean())


def depth_metrics(pred, gt, mask=None, align=False, eps=1e-6):
    """MAE / RMSE / AbsRel / delta<1.25, with optional scale+shift alignment."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(gt) & (gt > 0)
    p, g = pred[mask], gt[mask]
    if align:
        A = np.stack([p, np.ones_like(p)], axis=1)
        s, t = np.linalg.lstsq(A, g, rcond=None)[0]
        p = s * p + t
    diff = p - g
    ratio = np.maximum(p / (g + eps), g / (p + eps))
    return {
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "absrel": float((np.abs(diff) / (g + eps)).mean()),
        "delta1": float((ratio < 1.25).mean()),
    }
