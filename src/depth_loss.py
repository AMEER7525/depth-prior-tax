"""Depth supervision losses, selected by config.

  - "ssi":      scale-and-shift-invariant (MiDaS-style, closed-form alignment)
  - "pearson":  scale/shift-invariant correlation loss (no explicit alignment)
  - "absolute": direct L1 (assumes metric-aligned depth)

All functions take a single image's depth maps of shape (H, W) or flat (N,),
with an optional boolean validity mask.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def align_scale_shift(pred, target, mask=None, eps=1e-6):
    """Closed-form least squares for (s, t) minimizing ||s*pred + t - target||^2."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    n = pred.numel()
    sx, sxx = pred.sum(), (pred * pred).sum()
    sy, sxy = target.sum(), (pred * target).sum()
    det = torch.clamp(sxx * n - sx * sx, min=eps)
    s = (sxy * n - sx * sy) / det
    t = (sxx * sy - sx * sxy) / det
    return s, t


def ssi_loss(pred, target, mask=None, robust="l1"):
    """Scale-and-shift-invariant loss: align pred to target, then penalize residual."""
    s, t = align_scale_shift(pred, target, mask)
    aligned = s * pred + t
    if mask is not None:
        aligned, target = aligned[mask], target[mask]
    diff = aligned - target
    if robust == "l1":
        return diff.abs().mean()
    if robust == "huber":
        return F.huber_loss(aligned, target)
    return (diff ** 2).mean()


def pearson_loss(pred, target, mask=None, eps=1e-6):
    """1 - Pearson correlation; invariant to scale and shift by construction."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    pred = pred.reshape(-1) - pred.mean()
    target = target.reshape(-1) - target.mean()
    corr = (pred * target).sum() / (pred.norm() * target.norm() + eps)
    return 1.0 - corr


def absolute_loss(pred, target, mask=None):
    """Direct L1 on metric depth (no alignment)."""
    diff = (pred - target).abs()
    if mask is not None:
        diff = diff[mask]
    return diff.mean()


DEPTH_LOSSES = {"ssi": ssi_loss, "pearson": pearson_loss, "absolute": absolute_loss}


def depth_loss(pred, target, kind="ssi", mask=None, **kwargs):
    """Dispatch to the configured depth loss."""
    if kind not in DEPTH_LOSSES:
        raise ValueError(f"unknown depth loss '{kind}', choose from {list(DEPTH_LOSSES)}")
    return DEPTH_LOSSES[kind](pred, target, mask=mask, **kwargs)
