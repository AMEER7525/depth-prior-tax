"""Depth Anything V2: the one real monocular operating point (stage4_real).

WHAT THIS MODEL ACTUALLY OUTPUTS, and why it matters here more than usual.

DAv2 predicts *relative inverse depth* (disparity-like), defined only up to an
unknown scale and shift. It is not metric depth. That ambiguity is not an
inconvenience to paper over -- it is the exact phenomenon this project studies.
Two consequences shape every use of this module:

  - The unknown affine lives in INVERSE-depth space. 1/disparity is not an
    affine function of depth when the shift is non-zero, so alignment is done
    on disparity (align_inverse_depth / align_inverse_to_points) and inverted
    afterwards, never on 1/disparity.

  - As an INIT source, backprojection needs METRIC depth. The raw prediction
    backprojects to garbage geometry. It must first be aligned to something
    with real scale: GT depth (an oracle -- isolates the model's non-affine
    error) or SfM points (what a practitioner can actually do). The (scale,
    shift) solved are returned so callers can log them rather than discard them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DA_V2_REPOS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",   #   99 MB
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",    #  390 MB
    "large": "depth-anything/Depth-Anything-V2-Large-hf",   # 1341 MB
}
DEFAULT_VARIANT = "large"     # what the 3DGS depth-regularization docs use


def pick_device(prefer=None):
    """cuda > mps > cpu. MPS works, so depth can be precomputed on the Mac."""
    import torch
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_depth_model(variant=DEFAULT_VARIANT, local_dir=None, device=None):
    """Load DAv2 from a local directory if given, else from the Hub.

    Returns (model, processor, device). Pass local_dir to run from weights
    already on disk (Drive, or a cached copy) instead of hitting the Hub.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if variant not in DA_V2_REPOS:
        raise ValueError(f"variant must be one of {list(DA_V2_REPOS)}, got {variant!r}")
    source = local_dir if local_dir else DA_V2_REPOS[variant]
    device = pick_device(device)

    processor = AutoImageProcessor.from_pretrained(source)
    model = AutoModelForDepthEstimation.from_pretrained(source).to(device).eval()
    return model, processor, device


def predict_disparity(model, processor, image, device=None):
    """Run DAv2 on one HxWx3 image in [0,1]; return HxW relative INVERSE depth.

    Larger values are nearer. This is not metric and not depth -- see the module
    docstring before using it for anything but a scale-invariant loss.
    """
    import torch

    device = device or next(model.parameters()).device
    h, w = image.shape[:2]
    img_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    inputs = processor(images=img_u8, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth       # (1, h', w'), disparity-like

    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1).float(), size=(h, w), mode="bicubic", align_corners=False)
    return pred.squeeze().detach().cpu().numpy()


class CachedPredictor:
    """Disparity per view, computed once and kept on disk.

    The model (1.3 GB for 'large') is only loaded on the first cache miss, so a
    sweep re-running the same input views pays for inference once.
    """

    def __init__(self, cache_dir, variant=DEFAULT_VARIANT, local_dir=None, device=None):
        self.cache_dir = Path(cache_dir)
        self.variant, self.local_dir, self.device = variant, local_dir, device
        self._model = None
        self.hits = self.misses = 0

    def __call__(self, view):
        path = self.cache_dir / f"{view.name}.npy"
        if path.exists():
            self.hits += 1
            return np.load(path)
        if self._model is None:
            self._model = load_depth_model(self.variant, self.local_dir, self.device)
        model, proc, dev = self._model
        disp = predict_disparity(model, proc, view.image, dev).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, disp)
        self.misses += 1
        return disp


def disparity_to_depth(disparity, eps=1e-6):
    """Invert disparity into *relative* depth. Still not metric."""
    d = np.asarray(disparity, dtype=np.float64)
    return 1.0 / np.maximum(d, eps)


def _fit_affine(x, y):
    A = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(s), float(t)


def align_to_metric(pred, reference, mask=None, eps=1e-8):
    """Least-squares fit s*pred + t to a metric reference, in the SAME space.

    Returns (aligned, scale, shift). Log the scale and shift: they are how far
    the prediction sits from correct scale.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs reference {ref.shape}")

    valid = np.isfinite(pred) & np.isfinite(ref) & (ref > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    if valid.sum() < 2:
        raise ValueError("need at least 2 valid pixels to solve for scale and shift")

    s, t = _fit_affine(pred[valid], ref[valid])
    if abs(s) < eps:
        raise ValueError("degenerate alignment: prediction has no variation")
    return s * pred + t, s, t


def _invert(inv_depth, eps=1e-8):
    """Inverse depth -> depth; non-positive inverse depth becomes invalid (0)."""
    inv = np.asarray(inv_depth, dtype=np.float64)
    return np.where(inv > eps, 1.0 / np.maximum(inv, eps), 0.0).astype(np.float32)


def align_inverse_depth(disparity, ref_depth, mask=None):
    """Fit s*disparity + t to 1/ref_depth on valid pixels; return (depth, s, t).

    The oracle alignment: per-image, against GT. What remains after it is the
    model's non-affine error, which is what the noise arm of Axis C emulates.
    """
    ref = np.asarray(ref_depth, dtype=np.float64)
    inv_ref = np.where(ref > 0, 1.0 / np.where(ref > 0, ref, 1.0), 0.0)
    _, s, t = align_to_metric(disparity, inv_ref,
                              mask=(ref > 0) if mask is None else (mask & (ref > 0)))
    return _invert(s * np.asarray(disparity, dtype=np.float64) + t), s, t


ALIGN_MIN_POINTS = 3       # below this the affine fit has no redundancy left


def align_inverse_to_points(disparity, px, z, min_points=1):
    """Fit s*disparity + t to 1/z at sparse metric points; return (depth, s, t).

    With >= 3 points this is least squares plus one MAD-based outlier
    rejection pass. Sparse-view SfM routinely gives fewer: 3-view
    NeRF-Synthetic triangulates 6-10 points in TOTAL. So with 2 points the
    affine fit is solved exactly, with 1 only a scale (shift 0), and a
    degenerate fit (non-finite or non-positive scale) falls back to
    scale-only. How well a practitioner can make monocular depth metric from
    3-view SfM is the measurement here -- not a failure to be raised. Only an
    empty view raises.
    """
    disp = np.asarray(disparity, dtype=np.float64)
    H, W = disp.shape
    px = np.asarray(px, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if len(z) < max(1, min_points):
        raise ValueError(f"only {len(z)} SfM point(s) in view (need >= "
                         f"{max(1, min_points)}) to align monocular depth")
    j = np.clip(np.floor(px[:, 0]).astype(int), 0, W - 1)
    i = np.clip(np.floor(px[:, 1]).astype(int), 0, H - 1)
    x, y = disp[i, j], 1.0 / z

    def scale_only():
        return float(np.sum(x * y) / max(float(np.sum(x * x)), 1e-12)), 0.0

    if len(z) >= ALIGN_MIN_POINTS:
        s, t = _fit_affine(x, y)
        r = s * x + t - y
        mad = np.median(np.abs(r - np.median(r)))
        if mad > 0:
            keep = np.abs(r - np.median(r)) < 3 * 1.4826 * mad
            if keep.sum() >= 2:
                s, t = _fit_affine(x[keep], y[keep])
    elif len(z) == 2 and abs(x[1] - x[0]) > 1e-9:
        s, t = _fit_affine(x, y)          # exactly determined
    else:
        s, t = scale_only()
    if not np.isfinite(s) or s <= 0:
        s, t = scale_only()
    return _invert(s * disp + t), s, t
