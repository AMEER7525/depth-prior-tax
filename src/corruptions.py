"""Controlled degradation of ground-truth depth (Axis C).

Additive noise, a global affine (scale + shift) bias, and a per-image affine
bias turn 'prior quality' into reproducible knobs:

    d' = (scale * a_i) * (d + n) + (shift + b_i)       on valid pixels only

  n    zero-mean Gaussian noise, std = sigma_rel * median(valid depth). i.i.d.
       per pixel, or low-frequency when corr_px > 0 -- monocular depth errors
       are spatially smooth, and i.i.d. noise mostly averages out.
  a_i  per-image scale, log-normal with log-std scale_jitter
  b_i  per-image shift, std shift_jitter * median(valid depth)

Invalid pixels (depth <= 0 or non-finite: background, no return) stay
invalid. Every consumer -- backprojection, the depth-loss mask, the metrics --
reads depth > 0 as "valid", so perturbing a background zero would conjure
points out of empty space. For the same reason the median is taken over valid
pixels only: NeRF-Synthetic views are mostly background, and a median over all
pixels is 0, which would make every noise level a silent no-op.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _valid(depth):
    return torch.isfinite(depth) & (depth > 0)


def _valid_median(depth):
    v = _valid(depth)
    if not v.any():
        return None
    return depth[v].median().clamp(min=1e-6)


def _smooth_unit_noise(noise, corr_px):
    """Gaussian-blur a noise field and renormalize it to unit std."""
    r = max(1, int(math.ceil(3 * corr_px)))
    x = torch.arange(-r, r + 1, dtype=noise.dtype, device=noise.device)
    k = torch.exp(-x ** 2 / (2 * corr_px ** 2))
    k = k / k.sum()
    n = noise[None, None]
    n = F.conv2d(F.pad(n, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    n = F.conv2d(F.pad(n, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    n = n[0, 0]
    return n / n.std().clamp(min=1e-12)


def add_gaussian_noise(depth, sigma_rel=0.05, generator=None, corr_px=0.0):
    """Zero-mean noise with std = sigma_rel * median(valid depth), valid pixels only."""
    med = _valid_median(depth)
    if med is None:
        return depth.clone()
    noise = torch.randn(depth.shape, generator=generator, dtype=depth.dtype)
    noise = noise.to(depth.device)
    if corr_px > 0:
        noise = _smooth_unit_noise(noise, corr_px)
    return torch.where(_valid(depth), depth + noise * (sigma_rel * med), depth)


def affine_bias(depth, scale=1.0, shift=0.0):
    """scale * depth + shift on valid pixels (the monocular failure mode)."""
    return torch.where(_valid(depth), scale * depth + shift, depth)


def degrade_view(depth, sigma_rel=0.0, scale=1.0, shift=0.0, scale_jitter=0.0,
                 shift_jitter=0.0, corr_px=0.0, generator=None,
                 jitter_generator=None):
    """Degrade one view's depth. Returns (degraded, applied).

    `applied` records the scale/shift actually used for this view, so a
    per-image arm can be audited after the fact. Noise and jitter draw from
    separate generators, so turning one knob on does not reshuffle the other's
    random draws -- the noise pattern at sigma_rel=0.1 and 0.2 is the same
    field at two amplitudes.
    """
    med = _valid_median(depth)
    med_f = float(med) if med is not None else 1.0
    out = depth
    if sigma_rel > 0:
        out = add_gaussian_noise(out, sigma_rel, generator, corr_px)

    jg = jitter_generator if jitter_generator is not None else generator
    a, b = float(scale), float(shift)
    if scale_jitter > 0:
        a *= math.exp(scale_jitter * torch.randn((), generator=jg).item())
    if shift_jitter > 0:
        b += shift_jitter * med_f * torch.randn((), generator=jg).item()
    if a != 1.0 or b != 0.0:
        out = affine_bias(out, a, b)
    return out, {"scale": a, "shift": b, "noise_std": sigma_rel * med_f}


def degrade(depth, sigma_rel=0.0, scale=1.0, shift=0.0, generator=None, **kw):
    """degrade_view without the bookkeeping. Identity under defaults."""
    return degrade_view(depth, sigma_rel, scale, shift, generator=generator, **kw)[0]
