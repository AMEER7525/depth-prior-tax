"""Depth Anything V2: the one real monocular operating point (stage4_real).

WHAT THIS MODEL ACTUALLY OUTPUTS, and why it matters here more than usual.

DAv2 predicts *relative inverse depth* (disparity-like), defined only up to an
unknown scale and shift. It is not metric depth. That ambiguity is not an
inconvenience to paper over -- it is the exact phenomenon this project studies.
Two consequences shape every use of this module:

  - As a LOSS target, feed the raw prediction to a scale-and-shift-invariant
    loss, which is built for it. Feeding it to the absolute loss compares
    disparity against metres and is meaningless.

  - As an INIT source, backprojection needs METRIC depth. The raw prediction
    backprojects to garbage geometry. It must first be aligned to something
    with real scale (SfM points, or GT where available). The alignment residual
    is the real model's position on the controlled-degradation curve -- so
    align_to_metric() reports the (scale, shift) it solved, and callers should
    log it rather than discard it.
"""
from __future__ import annotations

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


def disparity_to_depth(disparity, eps=1e-6):
    """Invert disparity into *relative* depth. Still not metric."""
    d = np.asarray(disparity, dtype=np.float64)
    return 1.0 / np.maximum(d, eps)


def align_to_metric(pred, reference, mask=None, eps=1e-8):
    """Least-squares fit s*pred + t to a metric reference.

    Returns (aligned, scale, shift). Log the scale and shift: they are how far
    the real model sits from correct scale, which is what places it on the
    synthetic degradation curve from Axis C.
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

    p, r = pred[valid], ref[valid]
    A = np.stack([p, np.ones_like(p)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, r, rcond=None)
    if abs(s) < eps:
        raise ValueError("degenerate alignment: prediction has no variation")
    return s * pred + t, float(s), float(t)
