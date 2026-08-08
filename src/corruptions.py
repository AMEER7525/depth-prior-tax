"""Controlled degradation of ground-truth depth (Axis C).

Additive noise and/or a global affine (scale + shift) bias turn 'prior quality'
into a reproducible knob.
"""
from __future__ import annotations
import torch


def add_gaussian_noise(depth, sigma_rel=0.05, generator=None):
    """Zero-mean Gaussian noise with std = sigma_rel * median(depth)."""
    scale = depth.median().clamp(min=1e-6)
    noise = torch.randn(depth.shape, generator=generator, device=depth.device)
    return depth + noise * (sigma_rel * scale)


def affine_bias(depth, scale=1.0, shift=0.0):
    """Global affine distortion: scale * depth + shift (the monocular failure mode)."""
    return scale * depth + shift


def degrade(depth, sigma_rel=0.0, scale=1.0, shift=0.0, generator=None):
    """Compose noise then affine bias. Identity under defaults."""
    out = depth
    if sigma_rel > 0:
        out = add_gaussian_noise(out, sigma_rel, generator)
    if scale != 1.0 or shift != 0.0:
        out = affine_bias(out, scale, shift)
    return out
